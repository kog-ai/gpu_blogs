#include <cstdio>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "common.hpp"

constexpr auto num_cu_per_xcd = 38, num_xcd = 8;

constexpr auto num_waves_per_cu = 16,   // Tweak this value to adjust the number of waves resident on each CU
               num_waves = num_cu_per_xcd * num_xcd * num_waves_per_cu;

__global__
void kern_permute_lcy(std::uint64_t *latencies) {
    auto bid = blockIdx.x, wid = threadIdx.y, lid = threadIdx.x;

    auto src_lane = (lid + 1) & (__builtin_amdgcn_wavefrontsize() - 1);
    auto addr = src_lane << 2, val = (wid << 16) | lid;

    __builtin_amdgcn_s_barrier();

    __builtin_amdgcn_sched_barrier(0);
    auto start = __builtin_amdgcn_s_memtime();
    __builtin_amdgcn_sched_barrier(0);

    // The optimization barrier forces the emission of a waitcnt
    auto shuffled = __builtin_amdgcn_ds_permute(addr, val);
    asm volatile("" : "=v"(shuffled) : "0"(shuffled));

    __builtin_amdgcn_sched_barrier(0);
    auto end = __builtin_amdgcn_s_memtime();
    __builtin_amdgcn_sched_barrier(0);

    if (lid == 0)
        latencies[bid * num_waves_per_cu + wid] = end - start;
}

int main() {
    std::uint64_t *latencies;
    HIP_CHECK(hipMalloc(&latencies,   num_waves * sizeof(*latencies)));
    HIP_CHECK(hipMemset(latencies, 0, num_waves * sizeof(*latencies)));

    kern_permute_lcy<<<num_cu_per_xcd * num_xcd, dim3(64, num_waves_per_cu)>>>(latencies);
    HIP_CHECK(hipDeviceSynchronize());

    std::printf("ds_permute with %d wave%s:\n", num_waves_per_cu, num_waves_per_cu > 1 ? "s" : "");
    for (int i = 0; i < num_waves_per_cu; ++i)
        std::printf("  wave %2d: %5lu clk\n", i, latencies[i]);

    return 0;
}
