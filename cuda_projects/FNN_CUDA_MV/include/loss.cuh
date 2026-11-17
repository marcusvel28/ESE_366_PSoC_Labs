#ifndef LOSS_CUH
#define LOSS_CUH

#include <cuda_runtime.h>

// Output-layer gradient: delta = probabilities - targets
__global__ void output_delta(
    const float* probabilities,
    const float* targets,
    float* delta,
    int n
);

// Cross-entropy loss per sample
__global__ void cross_entropy(
    const float* probabilities,
    const float* targets,
    float* losses,
    int rows,
    int cols
);

#endif
