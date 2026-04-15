#pragma once

#include <cstdio>

#include <hip/hip_runtime.h>

#define HIP_CHECK(call)                                                        \
    do {                                                                       \
        const hipError_t err = (call);                                         \
        if (err != hipSuccess) {                                               \
            std::fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, hipGetErrorString(err));                    \
            return 1;                                                          \
        }                                                                      \
    } while (0)