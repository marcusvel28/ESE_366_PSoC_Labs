#ifndef ACTIVATION_CUH
#define ACTIVATION_CUH

#include <cuda_runtime.h>

// ReLU activation
__global__ void relu(float* y, const float* z, int n);

// ReLU derivative applied to a gradient
__global__ void relu_backward(
    float* delta, const float* activation, int n
);

// Stable row-wise softmax
__global__ void softmax(
    const float* logits, float* probabilities,
    int rows, int cols
);

#endif