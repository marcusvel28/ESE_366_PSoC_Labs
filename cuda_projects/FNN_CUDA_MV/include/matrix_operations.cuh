#ifndef MATRIX_OPERATIONS_CUH
#define MATRIX_OPERATIONS_CUH

#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// Matrix multiplication mode
enum CUDA_Mode
{
    CUDA_BASELINE,
    CUDA_OPTIMIZED
};

// C = A * B
// Baseline: direct global-memory access
__global__ void MatrixMultiplication_Baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// C = A * B
// Optimized: shared-memory tiling and manual loop unrolling
__global__ void MatrixMultiplication_Optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// B = A^T
__global__ void MatrixTranspose(
    const float* A, float* B,
    int rows, int cols
);

// Z = Z + b
__global__ void AddBias(
    float* Z, const float* b,
    int rows, int cols
);

// C = A + B
__global__ void MatrixAddition(
    const float* A, const float* B, float* C,
    int n
);

#endif
