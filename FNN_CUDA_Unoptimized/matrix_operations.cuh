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
// Settings (pass with -D on the nvcc line)
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
// Kernels (each does ONE small job, so every intermediate result goes through global memory)
// ---------------------------------------------------------------------------

// C = A * B.  A is MxK, B is KxN, C is MxN. No tiling: every operand is read from global memory.
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K);

// Z[r][c] += b[c]
__global__ void add_bias(float *Z, const float *b, int M, int N);

// Z[i] = max(Z[i], 0)   (flat, n elements)
__global__ void relu_activation(float *Z, int n);

// p = softmax(z) for each row. One thread per row.
__global__ void softmax_rows(const float *z, float *p, int M, int C);

// delta = (p - onehot(label)) / M, in place; also adds the loss. One thread per row.
__global__ void output_delta(float *delta, const int *labels, float *loss, int M, int C);

// B[cols x rows] = A[rows x cols] transposed (no tiling: writes are strided)
__global__ void matrix_transpose(const float *A, float *B, int rows, int cols);

// db[c] = sum of D down column c
__global__ void col_sum(const float *D, float *db, int M, int N);

// d[i] = 0 wherever act[i] <= 0   (ReLU derivative applied as its own pass)
__global__ void relu_mask(float *d, const float *act, int n);

// w[i] -= lr * g[i]
__global__ void sgd_update(float *w, const float *g, float lr, int n);

// pred[r] = index of the largest value in row r. One thread per row.
__global__ void argmax_rows(const float *a, int *pred, int M, int C);

// Cache-config hints
void set_cache_preferences();

#endif // MATRIX_OPERATIONS_CUH
