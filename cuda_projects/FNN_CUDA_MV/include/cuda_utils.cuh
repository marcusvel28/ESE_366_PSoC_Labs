#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Check CUDA API calls and report errors
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__,                       \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

#endif