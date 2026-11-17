#include "matrix_operations.cuh"

// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// Baseline: direct global-memory access
__global__ void MatrixMultiplication_Baseline(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M)
    {
        float sum = 0.0f;

        for (int k = 0; k < K; k++)
        {
            sum += A[y * K + k] * B[k * N + x];
        }

        C[y * N + x] = sum;
    }
}


// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
// Optimized: 16x16 shared-memory tiling and loop unrolling
__global__ void MatrixMultiplication_Optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)
    {
        int colA = t * BLOCK_SIZE + threadIdx.x;
        int rowB = t * BLOCK_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (y < M && colA < K) ? A[y * K + colA] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (rowB < K && x < N) ? B[rowB * N + x] : 0.0f;

        __syncthreads();

        // Manual unrolling (factor of 4)
        for (int k = 0; k < BLOCK_SIZE; k += 4)
        {
            sum += tileA[threadIdx.y][k]     * tileB[k][threadIdx.x];
            sum += tileA[threadIdx.y][k + 1] * tileB[k + 1][threadIdx.x];
            sum += tileA[threadIdx.y][k + 2] * tileB[k + 2][threadIdx.x];
            sum += tileA[threadIdx.y][k + 3] * tileB[k + 3][threadIdx.x];
        }

        __syncthreads();
    }

    if (x < N && y < M)
    {
        C[y * N + x] = sum;
    }
}


// Matrix transpose: B = A^T
__global__ void MatrixTranspose(
    const float* A, float* B,
    int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        B[x * rows + y] = A[y * cols + x];
    }
}


// Bias addition: Z = Z + b
__global__ void AddBias(
    float* Z, const float* b,
    int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows)
    {
        Z[y * cols + x] += b[x];
    }
}


// Matrix addition: C = A + B
__global__ void MatrixAddition(
    const float* A, const float* B, float* C,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}
