#ifndef MATRIX_OPERATIONS_CUH
#define MATRIX_OPERATIONS_CUH

#include <cuda_runtime.h>

// Available CUDA modes
enum CUDA_Mode
{
    CUDA_BASELINE,
    CUDA_OPTIMIZED
};

// Baseline matrix multiplication
__global__ void MatrixMultiplication_Baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// Optimized matrix multiplication
template<int TILE>
__global__ void MatrixMultiplication_Optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K
);

// Matrix multiplication launcher
void LaunchMatrixMultiplication(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    CUDA_Mode mode, int block_size
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
