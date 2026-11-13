//**************************************************************************
//
// File Name : "matrix_operations.cuh"
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
#ifndef MATRIX_OPERATIONS_CUH
#define MATRIX_OPERATIONS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Settings (change these for experiments, or pass them with -D on the nvcc line)
// ---------------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16      // matmul / transpose block is BLOCK_SIZE x BLOCK_SIZE threads
#endif
#ifndef UNROLL
#define UNROLL 4           // unroll factor of the matmul inner loop
#endif
#define THREADS_1D 256     // threads per block for the 1-D kernels

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
    exit(1); } } while (0)

#define GRID1D(n) (((n) + THREADS_1D - 1) / THREADS_1D)

// 2-D grid that covers a matrix with `rows` rows and `cols` columns
static inline dim3 grid2D(int cols, int rows)
{
    return dim3((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// out = relu(in * W + b).  in is MxK, W is KxN, out is MxN.   (bias + ReLU fused)
__global__ void forward(const float * __restrict__ in, const float * __restrict__ W,
                        const float * __restrict__ b, float *out,
                        int M, int N, int K, int apply_relu);

// Softmax + cross-entropy delta + loss. One thread per image.
__global__ void softmax_delta(const float *z, const int *labels, float *delta,
                              float *loss, int M, int C);

// B[cols x rows] = A[rows x cols] transposed (shared-memory tiled)
__global__ void matrix_transpose(const float * __restrict__ A, float *B, int rows, int cols);

// W -= lr * (inT * D).  inT is MxK, D is KxN, W is MxN.   (gradient + update fused)
__global__ void weight_grad_update(const float * __restrict__ inT, const float * __restrict__ D,
                                   float *W, float lr, int M, int N, int K);

// b -= lr * (sum of D down each column)
__global__ void col_sum_update(const float * __restrict__ D, float *b, float lr, int M, int N);

// Dprev = (D * WT) masked by relu'(act)
__global__ void delta_backward(const float * __restrict__ D, const float * __restrict__ WT,
                               float *Dprev, const float * __restrict__ act,
                               int M, int N, int K);

// pred[r] = index of the largest value in row r. One thread per row.
__global__ void argmax_rows(const float *a, int *pred, int M, int C);

// Cache-config hints for the matmul kernels (try PreferL1 / PreferShared / PreferNone)
void set_cache_preferences();

#endif // MATRIX_OPERATIONS_CUH
