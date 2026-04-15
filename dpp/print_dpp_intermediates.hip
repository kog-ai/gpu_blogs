#include <cstdint>
#include <cstdio>
#include <vector>

#include <hip/hip_runtime.h>

#include "common.hpp"
#include "dpp.hpp"

constexpr int num_waves = 1;
constexpr int wave_size = 64;
constexpr int num_lanes = num_waves * wave_size;
constexpr int step_count = 8;

constexpr const char *step_names[step_count] = {
    "original", "shr1",    "shr2",    "shr4",
    "shr8",     "bcast15", "bcast31", "readlane63",
};

__device__ __forceinline__ void dpp_wave_sum_store_steps(std::uint32_t val,
                                                         std::uint32_t *steps) {
    steps[0] = val;
    val += dpp<0x111>(val); // row_shr:1
    steps[1] = val;
    val += dpp<0x112>(val); // row_shr:2
    steps[2] = val;
    val += dpp<0x114>(val); // row_shr:4
    steps[3] = val;
    val += dpp<0x118>(val); // row_shr:8
    steps[4] = val;
    val += dpp<0x142>(val); // row_bcast:15
    steps[5] = val;
    val += dpp<0x143>(val); // row_bcast:31
    steps[6] = val;
    steps[7] = dpp_wave_broadcast63(steps[6]);
}

__global__ void capture_wave_sum_step_by_step_kernel(const std::uint32_t *input,
                                                     std::uint32_t *all_steps) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_lanes) {
        return;
    }

    std::uint32_t steps[step_count];
    dpp_wave_sum_store_steps(input[idx], steps);

#pragma unroll
    for (int step = 0; step < step_count; ++step) {
        all_steps[step * num_lanes + idx] = steps[step];
    }
}

void print_saved_lane_steps(const std::vector<std::uint32_t> &all_steps) {
    std::printf("Showing wave reduction steps for each lane:\n\n");
    for (int step = 0; step < step_count; ++step) {
        std::printf("%s\n[", step_names[step]);
        for (int lane = 0; lane < wave_size; ++lane) {
            const int idx = lane;
            const std::uint32_t value = all_steps[step * num_lanes + idx];
            std::printf("%u", value);
            if (lane + 1 < wave_size) {
                std::printf(" ");
            }
        }
        std::printf("]\n\n");
    }
}

int run_print_dpp_intermediates() {
    std::vector<std::uint32_t> host_input(num_lanes);
    for (int wave = 0; wave < num_waves; ++wave) {
        for (int lane = 0; lane < wave_size; ++lane) {
            host_input[wave * wave_size + lane] =
                static_cast<std::uint32_t>(lane % 16);
        }
    }

    std::vector<std::uint32_t> host_steps(num_lanes * step_count, 0);

    std::uint32_t *device_input = nullptr;
    std::uint32_t *device_steps = nullptr;

    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_input),
                        host_input.size() * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(reinterpret_cast<void **>(&device_steps),
                        host_steps.size() * sizeof(std::uint32_t)));

    HIP_CHECK(hipMemcpy(device_input, host_input.data(),
                        host_input.size() * sizeof(std::uint32_t),
                        hipMemcpyHostToDevice));

    const dim3 grid((num_lanes + wave_size - 1) / wave_size);
    const dim3 block(wave_size);
    hipLaunchKernelGGL(capture_wave_sum_step_by_step_kernel, grid, block, 0, 0,
                       device_input, device_steps);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(host_steps.data(), device_steps,
                        host_steps.size() * sizeof(std::uint32_t),
                        hipMemcpyDeviceToHost));

    print_saved_lane_steps(host_steps);

    HIP_CHECK(hipFree(device_steps));
    HIP_CHECK(hipFree(device_input));
    return 0;
}

int main() { return run_print_dpp_intermediates(); }
