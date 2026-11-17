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

template<int TILE>
__global__ void MatrixMultiplication_Optimized(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++)
    {
        int colA = t * TILE + threadIdx.x;
        int rowB = t * TILE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (y < M && colA < K)
                ? A[y * K + colA] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] =
            (rowB < K && x < N)
                ? B[rowB * N + x] : 0.0f;

        __syncthreads();

        // Manually unrolled dot product (factor of 4)
        for (int k = 0; k < TILE; k += 4)
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


void LaunchMatrixMultiplication(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    CUDA_Mode mode, int block_size)
{
    dim3 block(block_size, block_size);

    dim3 grid(
        (N + block_size - 1) / block_size,
        (M + block_size - 1) / block_size
    );

    if (mode == CUDA_BASELINE)
    {
        MatrixMultiplication_Baseline<<<grid, block>>>(
            A, B, C, M, N, K
        );
    }
    else
    {
        switch (block_size)
        {
            case 8:
                MatrixMultiplication_Optimized<8>
                    <<<grid, block>>>(A, B, C, M, N, K);
                break;

            case 16:
                MatrixMultiplication_Optimized<16>
                    <<<grid, block>>>(A, B, C, M, N, K);
                break;

            case 32:
                MatrixMultiplication_Optimized<32>
                    <<<grid, block>>>(A, B, C, M, N, K);
                break;
        }
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
