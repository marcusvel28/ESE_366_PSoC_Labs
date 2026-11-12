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

#ifdef FAST_MATH           // use the Special Function Units for exp
#define EXP(x) __expf(x)
#define LOG(x) __logf(x)
#else
#define EXP(x) expf(x)
#define LOG(x) logf(x)
#endif

// ---- Naive matrix multiply: C[M x N] = A[M x K] * B[K x N] ------------------
// One thread per output element. Each step of the k loop reads A and B from global
// memory. Only the running sum is in a register.
__global__ void matmul(const float *A, const float *B, float *C, int M, int N, int K)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;   // column of C
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;   // row of C

    if (ix < N && iy < M)
    {
        float sum = 0.0f;
        #pragma unroll UNROLL
        for (int k = 0; k < K; k++)
            sum += A[iy * K + k] * B[k * N + ix];
        C[iy * N + ix] = sum;
    }
}

// ---- Bias add: separate pass over the matrix --------------------------------
__global__ void add_bias(float *Z, const float *b, int M, int N)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < N && iy < M) Z[iy * N + ix] += b[ix];
}

// ---- ReLU: another separate pass --------------------------------------------
__global__ void relu_activation(float *Z, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) Z[i] = fmaxf(Z[i], 0.0f);
}

// ---- Softmax: writes the probabilities to global memory ----------------------
__global__ void softmax_rows(const float *z, float *p, int M, int C)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= M) return;

    float m = z[r * C];
    for (int c = 1; c < C; c++) m = fmaxf(m, z[r * C + c]);
    float s = 0.0f;
    for (int c = 0; c < C; c++) s += EXP(z[r * C + c] - m);
    for (int c = 0; c < C; c++) p[r * C + c] = EXP(z[r * C + c] - m) / s;
}

// ---- Output delta + loss: a second pass that reads the probabilities back ----
__global__ void output_delta(float *delta, const int *labels, float *loss, int M, int C)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= M) return;

    int label = labels[r];
    for (int c = 0; c < C; c++)
    {
        float p = delta[r * C + c];
        delta[r * C + c] = (p - (c == label ? 1.0f : 0.0f)) / M;
        if (c == label) atomicAdd(loss, -LOG(fmaxf(p, 1e-12f)));
    }
}

// ---- Naive transpose: reads are coalesced, writes are strided ----------------
__global__ void matrix_transpose(const float *A, float *B, int rows, int cols)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;   // column of A
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;   // row of A
    if (ix < cols && iy < rows)
        B[ix * rows + iy] = A[iy * cols + ix];
}

// ---- Bias gradient: written to global memory ---------------------------------
__global__ void col_sum(const float *D, float *db, int M, int N)
{
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (c >= N) return;
    float s = 0.0f;
    for (int r = 0; r < M; r++) s += D[r * N + c];
    db[c] = s;
}

// ---- ReLU derivative: its own pass over the delta matrix ---------------------
__global__ void relu_mask(float *d, const float *act, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n && act[i] <= 0.0f) d[i] = 0.0f;
}

// ---- Weight / bias update: reads the stored gradient back --------------------
__global__ void sgd_update(float *w, const float *g, float lr, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) w[i] -= lr * g[i];
}

// ---- Prediction --------------------------------------------------------------
__global__ void argmax_rows(const float *a, int *pred, int M, int C)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= M) return;
    int best = 0;
    for (int c = 1; c < C; c++) if (a[r * C + c] > a[r * C + best]) best = c;
    pred[r] = best;
}

// ---- Cache-config hint -------------------------------------------------------
// Ask for the larger L1 split. (The naive matmul uses no shared memory.)
void set_cache_preferences()
{
    cudaFuncSetCacheConfig(matmul, cudaFuncCachePreferL1);
}
