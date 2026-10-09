#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

#include <fmt/core.h>

#include "calc.h"
#include "cmp.h"
#include "error.h"
#include "timer.h"

// C[M, N] = A[M, K] * B[K, N], all row-major. Every dim is a multiple of TILE so
// the kernels below can stay branch-free on their hot paths.
constexpr auto M = 1024;
constexpr auto K = 1024;
constexpr auto N = 1024;
constexpr auto REPEAT_TIME = 4;

// fp32 matmul sums K = 1024 products, and the host and the device sum them in a
// different order, so the two results cannot match bit-for-bit: on the inputs
// built below they disagree by ~1e-4 absolute, on C values of ~260 (~4e-7
// relative). The 1e-6 absolute eps used by the elementwise ops would report a
// false mismatch here. 1e-3 keeps ~10x headroom over what is observed today,
// and the asymmetry grows roughly linearly with K, so revisit it if K changes.
constexpr auto EPS = 1e-3f;

namespace host {
// Reference implementation. The i-k-j loop order keeps the innermost loop
// walking contiguous memory in both b and c, which is what makes a plain triple
// loop fast enough to sit inside a REPEAT_TIME timing loop.
auto matmul_host(const std::vector<float>& a, const std::vector<float>& b)
    -> std::tuple<std::vector<float>, float> {
    auto c = std::vector<float>(M * N);
    auto elapsed = util::time_cpu(REPEAT_TIME, [&]() {
        std::fill(c.begin(), c.end(), 0.0f);  // c accumulates across k -> reset every repeat
        for (auto i = 0; i < M; ++i) {
            for (auto k = 0; k < K; ++k) {
                auto a_ik = a[i * K + k];
                for (auto j = 0; j < N; ++j) {
                    c[i * N + j] += a_ik * b[k * N + j];
                }
            }
        }
    });
    return {std::move(c), elapsed};
}
}  // namespace host

namespace device {
constexpr auto TILE = 32;  // 32x32 = 1024 threads = one full block
static_assert(M % TILE == 0 && N % TILE == 0 && K % TILE == 0, "M, N and K must be multiples of TILE");

template <typename Kernel>
auto matmul_device(Kernel kernel, dim3 grid, dim3 block, const float* a_d, const float* b_d, float* c_d)
    -> std::tuple<std::vector<float>, float> {
    auto elapsed = util::time_cuda(REPEAT_TIME, [&]() {
        kernel<<<grid, block>>>(a_d, b_d, c_d, M, N, K);
        CHECK_ERR(cudaGetLastError());
        CHECK_ERR(cudaDeviceSynchronize());
    });

    auto result = std::vector<float>(M * N);
    CHECK_ERR(cudaMemcpy(result.data(), c_d, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return {std::move(result), elapsed};
}

// Baseline: one thread per output element, each walking the full K dimension.
//
// The flops are not the problem here, the memory traffic is. Threads share
// nothing, so every element of A is re-loaded N times and every element of B is
// re-loaded M times: 2*M*N*K loads instead of the 2*M*K + 2*K*N that the two
// matrices actually contain -- 512x more at these sizes. The per-warp access
// pattern is already good -- a[row*k + i] is a single address broadcast to all 32
// lanes (row is constant across a warp) and b[i*n + col] is 32 consecutive
// addresses (coalesced) -- so L2 does absorb much of the redundancy (measured
// throughput is well above DRAM bandwidth), but not enough to win: the re-read
// factor of N and M is far larger than any cache.
__global__ void matmul_naive(const float* a, const float* b, float* c, int m, int n, int k) {
    const auto row = blockDim.y * blockIdx.y + threadIdx.y;
    const auto col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= m || col >= n) return;

    auto acc = 0.0f;
    for (auto i = 0; i < k; ++i) {
        acc = fmaf(a[row * k + i], b[i * n + col], acc);
    }
    c[row * n + col] = acc;
}

// Shared-memory tiling: one block owns a TILE x TILE tile of C and computes it
// while marching along K in TILE-wide slabs. For each slab the 1024 threads
// cooperatively load the matching TILE x TILE pieces of A and B into shared
// memory. Every value fetched from global memory is then read TILE times by the
// block rather than once, which is the whole point: global traffic drops by
// ~TILE and the repeated reads are served by shared memory.
//
// That said, each thread still produces exactly one output, so per iteration it
// issues two shared loads for one FMA, and its single accumulator is a serial
// dependency chain that stalls on each FMA's ~4-cycle latency. Tiling cut the
// global traffic, but the kernel is now issue/latency-bound, so it only beats
// the naive one by well under 2x (1.3-1.8x across measured runs) rather than by
// anything like TILE. Giving each thread a grid of accumulators (register
// tiling) is what fixes that.
__global__ void matmul_tiled(const float* a, const float* b, float* c, int m, int n, int k) {
    // 2 * 32*32 * 4B = 8 KB of shared memory per block, well under the 48 KB
    // default limit: at 38 registers per thread it is the 1024-thread block
    // size, not shared memory, that limits residency to one block per SM.
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const auto row = blockIdx.y * TILE + threadIdx.y;  // output row this thread is responsible for
    const auto col = blockIdx.x * TILE + threadIdx.x;  // output col this thread is responsible for
    auto acc = 0.0f;

    for (auto t = 0; t < k / TILE; ++t) {
        // Each thread contributes exactly one element to each slab. Both loads
        // are coalesced: consecutive threadIdx.x walks consecutive columns of A
        // (same row) and consecutive columns of B (same k).
        a_tile[threadIdx.y][threadIdx.x] = a[row * k + t * TILE + threadIdx.x];
        b_tile[threadIdx.y][threadIdx.x] = b[(t * TILE + threadIdx.y) * n + col];
        __syncthreads();  // tiles are only complete once every thread has loaded

        // The actual inner product for this thread's output element, read
        // entirely out of shared memory. Both reads are bank-conflict free:
        // a_tile[ty][i] is one address broadcast to the warp (ty is constant
        // within a warp) and b_tile[i][tx] is consecutive across the lanes.
        for (auto i = 0; i < TILE; ++i) {
            acc = fmaf(a_tile[threadIdx.y][i], b_tile[i][threadIdx.x], acc);
        }
        // Before the next slab overwrites the tiles, wait until every thread has
        // finished reading this one.
        __syncthreads();
    }

    c[row * n + col] = acc;
}
}  // namespace device

int main() {
    auto a_h = std::vector<float>(M * K);
    auto b_h = std::vector<float>(K * N);
    // Cheap deterministic multiplicative hash -> values in [0, 1). Mixed enough
    // that the two summation orders above genuinely disagree, unlike inputs that
    // are exactly representable and would make any order bit-identical.
    for (auto i = 0; i < M * K; ++i) {
        a_h[i] = static_cast<float>((i * 2654435761u) % 1000) / 1000.0f;
    }
    for (auto i = 0; i < K * N; ++i) {
        b_h[i] = static_cast<float>((i * 40503u) % 1000) / 1000.0f;
    }

    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;
    CHECK_ERR(cudaMalloc((void**)&a_d, a_h.size() * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&b_d, b_h.size() * sizeof(float)));
    CHECK_ERR(cudaMalloc((void**)&c_d, M * N * sizeof(float)));
    CHECK_ERR(cudaMemcpy(a_d, a_h.data(), a_h.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_ERR(cudaMemcpy(b_d, b_h.data(), b_h.size() * sizeof(float), cudaMemcpyHostToDevice));

    auto [host_result, cpu_elapsed] = host::matmul_host(a_h, b_h);
    fmt::println("[host] elapsed_time={} ms", cpu_elapsed);

    const auto thread_per_block = 32;
    auto naive_block = dim3(thread_per_block, thread_per_block, 1);
    auto naive_grid = dim3(util::ceil_div(N, thread_per_block), util::ceil_div(M, thread_per_block), 1);

    auto [device_result_naive, naive_elapsed] =
        device::matmul_device(device::matmul_naive, naive_grid, naive_block, a_d, b_d, c_d);
    auto [naive_ok, naive_max_idx] = util::cmp_vec(device_result_naive, host_result, EPS);
    auto naive_max_diff = std::fabs(host_result[naive_max_idx] - device_result_naive[naive_max_idx]);
    fmt::println("[device::naive] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", naive_elapsed,
                host_result[naive_max_idx], device_result_naive[naive_max_idx], naive_max_diff);
    if (!naive_ok) return -1;

    auto tiled_block = dim3(device::TILE, device::TILE, 1);
    auto tiled_grid = dim3(N / device::TILE, M / device::TILE, 1);

    auto [device_result_tiled, tiled_elapsed] =
        device::matmul_device(device::matmul_tiled, tiled_grid, tiled_block, a_d, b_d, c_d);
    auto [tiled_ok, tiled_max_idx] = util::cmp_vec(device_result_tiled, host_result, EPS);
    auto tiled_max_diff = std::fabs(host_result[tiled_max_idx] - device_result_tiled[tiled_max_idx]);
    fmt::println("[device::tiled] elapsed_time={} ms, result=[{}, {}], diff={:.10e}", tiled_elapsed,
                host_result[tiled_max_idx], device_result_tiled[tiled_max_idx], tiled_max_diff);
    if (!tiled_ok) return -1;

    CHECK_ERR(cudaFree(a_d));
    CHECK_ERR(cudaFree(b_d));
    CHECK_ERR(cudaFree(c_d));

    return 0;
}
