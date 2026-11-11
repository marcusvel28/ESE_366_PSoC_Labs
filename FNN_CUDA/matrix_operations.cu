//**************************************************************************
//
// File Name : "matrix_operations.cu"
// Title : matrix_operations
// Date : 11/09/2026
// Version : 1.0
// Target Hardware : NVIDIA Jetson Orin Nano
// Author : Adam Roccanova
//
// Revision History : Initial version
//
//
//**************************************************************************
#include "matrix_operations.cuh"
#define BLOCK_SIZE 32

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, int n) 
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (ix < n) 
    {
        for (int iy = 0; iy < n; iy++) 
        {
            int idx = iy * n + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

__global__ void mulMatrixSharedUnroll(const float *MatA, const float *MatB, float *MatC, int M, int N, int K)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;   // column of C (0..N)
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;   // row of C    (0..M)
    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)      // was n, now K
    {
        unsigned int aCol = t * BLOCK_SIZE + threadIdx.x;            // k index into A
        unsigned int bRow = t * BLOCK_SIZE + threadIdx.y;            // k index into B

        tileA[threadIdx.y][threadIdx.x] = (iy < M && aCol < K) ? MatA[iy * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && ix < N) ? MatB[bRow * N + ix] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (ix < N && iy < M)
    {
        MatC[iy * N + ix] = sum;
    }
}

__global__ void matrix_transpose_2D2D (float *MatA, float *MatB, int rows, int cols)
{
   unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;    //column index
   unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;    //row index

    if (ix < cols && iy < rows)
    {
        MatB[ix*n + iy] = MatA[iy*n + ix];
    }
}

__global__ void addBias(float *Z, const float *b, int rows, int cols)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;   // column
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;   // row

    if (ix < cols && iy < rows)
        {
            Z[iy * cols + ix] += b[ix];
        }
}