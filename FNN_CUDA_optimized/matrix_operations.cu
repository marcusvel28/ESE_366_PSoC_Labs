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

#ifdef FAST_MATH           // use the Special Function Units for exp / log
#define EXP(x) __expf(x)
#define LOG(x) __logf(x)
#else
#define EXP(x) expf(x)
#define LOG(x) logf(x)
#endif

// ---- Tiled matrix multiply core:  C[M x N] = A[M x K] * B[K x N] ----------
// Returns the ONE output element C[iy][ix] that the calling thread owns.
// Every thread in the block must call it (it contains __syncthreads()).
// OPT 5: tiles live in shared memory, the running sum lives in a register.
// OPT 6: const __restrict__ lets the compiler use the read-only cache path.
__device__ float mulMatrixOnGPUSharedUnroll(const float * __restrict__ A,
                                            const float * __restrict__ B,
                                            int M, int N, int K)
{
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;   // column of C
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;   // row of C
    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; t++)
    {
        unsigned int aCol = t * BLOCK_SIZE + threadIdx.x;      // k index into A
        unsigned int bRow = t * BLOCK_SIZE + threadIdx.y;      // k index into B

        tileA[threadIdx.y][threadIdx.x] = (iy < M && aCol < K) ? A[iy * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && ix < N) ? B[bRow * N + ix] : 0.0f;
        __syncthreads();

        #pragma unroll UNROLL
        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    return sum;
}

// ---- Forward:  out = relu(in * W + b) ---------------------------------------
// OPT 1: bias and ReLU are applied to the register value before the single write.
__global__ void forward(const float * __restrict__ in, const float * __restrict__ W,
                        const float * __restrict__ b, float *out,
                        int M, int N, int K, int apply_relu)
{
    float sum = mulMatrixOnGPUSharedUnroll(in, W, M, N, K);

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < N && iy < M)
    {
        float v = sum + __ldg(&b[ix]);
        out[iy * N + ix] = apply_relu ? fmaxf(v, 0.0f) : v;
    }
}

// ---- Softmax + cross-entropy delta + loss, one thread per image -------------
// delta = (softmax - onehot(label)) / M.   OPT 7: one kernel, one pass over the logits.
__global__ void softmax_delta(const float *z, const int *labels, float *delta,
                              float *loss, int M, int C)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= M) return;

    const float *zr = z + r * C;
    float m = zr[0];
    for (int c = 1; c < C; c++) m = fmaxf(m, zr[c]);
    float s = 0.0f;
    for (int c = 0; c < C; c++) s += EXP(zr[c] - m);

    int label = labels[r];
    for (int c = 0; c < C; c++)
    {
        float p = EXP(zr[c] - m) / s;
        delta[r * C + c] = (p - (c == label ? 1.0f : 0.0f)) / M;
        if (c == label) atomicAdd(loss, -LOG(fmaxf(p, 1e-12f)));
    }
}

// ---- Tiled transpose: B[cols x rows] = A[rows x cols]^T ----------------------
// OPT 4: read a tile coalesced, flip it inside shared memory, write it coalesced.
// The +1 padding avoids shared-memory bank conflicts.
__global__ void matrix_transpose(const float * __restrict__ A, float *B, int rows, int cols)
{
    __shared__ float tile[BLOCK_SIZE][BLOCK_SIZE + 1];

    unsigned int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;    // column in A
    unsigned int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;    // row in A
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = A[y * cols + x];
    __syncthreads();

    unsigned int bx = blockIdx.y * BLOCK_SIZE + threadIdx.x;   // column in B
    unsigned int by = blockIdx.x * BLOCK_SIZE + threadIdx.y;   // row in B
    if (bx < rows && by < cols) B[by * rows + bx] = tile[threadIdx.x][threadIdx.y];
}

// ---- Weight gradient + update:  W -= lr * (in^T * delta) ---------------------
// OPT 3: the gradient never goes to global memory; it is applied straight to W.
// Called with inT (Kin x batch) and D = delta (batch x Nout), so the output is Kin x Nout.
__global__ void weight_grad_update(const float * __restrict__ inT, const float * __restrict__ D,
                                   float *W, float lr, int M, int N, int K)
{
    float sum = mulMatrixOnGPUSharedUnroll(inT, D, M, N, K);

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < N && iy < M) W[iy * N + ix] -= lr * sum;
}

// ---- Bias gradient + update:  b -= lr * (sum of delta down each column) ------
// OPT 3 again. Neighboring threads read neighboring columns, so reads are coalesced.
__global__ void col_sum_update(const float * __restrict__ D, float *b, float lr, int M, int N)
{
    int c = threadIdx.x + blockIdx.x * blockDim.x;
    if (c >= N) return;
    float s = 0.0f;
    for (int r = 0; r < M; r++) s += D[r * N + c];
    b[c] -= lr * s;
}

// ---- Delta for the previous layer:  Dprev = (D * W^T) masked by relu' --------
// OPT 2: the ReLU derivative (1 if activation > 0, else 0) is applied before the write.
__global__ void delta_backward(const float * __restrict__ D, const float * __restrict__ WT,
                               float *Dprev, const float * __restrict__ act,
                               int M, int N, int K)
{
    float sum = mulMatrixOnGPUSharedUnroll(D, WT, M, N, K);

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < N && iy < M)
        Dprev[iy * N + ix] = (__ldg(&act[iy * N + ix]) > 0.0f) ? sum : 0.0f;
}

// ---- Prediction: index of the largest output, one thread per image ----------
__global__ void argmax_rows(const float *a, int *pred, int M, int C)
{
    int r = threadIdx.x + blockIdx.x * blockDim.x;
    if (r >= M) return;
    int best = 0;
    for (int c = 1; c < C; c++) if (a[r * C + c] > a[r * C + best]) best = c;
    pred[r] = best;
}

// ---- Cache-config hints ------------------------------------------------------
// Hint only: prefer L1 over shared memory for the matmul kernels. Try changing this
// (cudaFuncCachePreferShared / PreferNone) and time the difference.
void set_cache_preferences()
{
    cudaFuncSetCacheConfig(forward,            cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(weight_grad_update, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(delta_backward,     cudaFuncCachePreferL1);
}
