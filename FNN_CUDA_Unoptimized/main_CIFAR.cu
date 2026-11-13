//**************************************************************************
//
// File Name : "main.cu"
// Title : main
// Date : 11/11/2026
// Version : 1.0
// Target Hardware : NVIDIA Jetson Orin Nano
// Author : Adam Roccanova
//
// Revision History : Initial version
//
//
//**************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "neural_network.cuh"

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
#define BATCH   100        // divides both 50000 and 10000, so every batch is full
#define HIDDEN  256
#define EPOCHS  20
#define LR      0.02f
#define DATA_DIR "data/cifar-10-binary_tar/cifar-10-batches-bin"
#define IMG_DIM 3072   // 32*32*3

// Reads one CIFAR-10 batch file (10000 records of: 1 label byte + 3072 pixel bytes)
// into x[offset ...] and y[offset ...]. Returns the number of images read.
static int load_cifar_batch(const char *name, float *x, int *y, int offset)
{
    char path[512];  snprintf(path, sizeof(path), "%s/%s", DATA_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }

    const int N = 10000;
    unsigned char rec[1 + IMG_DIM];
    for (int i = 0; i < N; i++)
    {
        if (fread(rec, 1, sizeof(rec), f) != sizeof(rec)) { fprintf(stderr, "short read in %s\n", path); exit(1); }
        y[offset + i] = rec[0];
        float *dst = x + (size_t)(offset + i) * IMG_DIM;
        for (int j = 0; j < IMG_DIM; j++) dst[j] = rec[1 + j] / 255.0f;
    }
    fclose(f);
    return N;
}

static float* to_gpu(const float *h, size_t count)
{
    float *d;
    CHECK(cudaMalloc(&d, count * sizeof(float)));
    CHECK(cudaMemcpy(d, h, count * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

static int* to_gpu(const int *h, size_t count)
{
    int *d;
    CHECK(cudaMalloc(&d, count * sizeof(int)));
    CHECK(cudaMemcpy(d, h, count * sizeof(int), cudaMemcpyHostToDevice));
    return d;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main()
{
    srand(1);

        // ---- load CIFAR-10, then put the whole dataset on the GPU once ---------
    const int dim = IMG_DIM;
    const int ntr = 50000, nte = 10000;

    float *h_xtr = (float*)malloc((size_t)ntr * dim * sizeof(float));
    int   *h_ytr = (int*)  malloc(ntr * sizeof(int));
    float *h_xte = (float*)malloc((size_t)nte * dim * sizeof(float));
    int   *h_yte = (int*)  malloc(nte * sizeof(int));

    char name[64];
    for (int i = 0; i < 5; i++)
    {
        snprintf(name, sizeof(name), "data_batch_%d.bin", i + 1);
        load_cifar_batch(name, h_xtr, h_ytr, i * 10000);
    }
    load_cifar_batch("test_batch.bin", h_xte, h_yte, 0);

    float *d_xtr = to_gpu(h_xtr, (size_t)ntr * dim);
    int   *d_ytr = to_gpu(h_ytr, ntr);
    float *d_xte = to_gpu(h_xte, (size_t)nte * dim);
    int   *d_pred;  CHECK(cudaMalloc(&d_pred, BATCH * sizeof(int)));

    // ---- build the network -------------------------------------------------
    int sizes[] = { dim, HIDDEN, 10 };
    Neural_Network nn = nn_create(sizes, 3, BATCH, LR);
    set_cache_preferences();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);  cudaEventCreate(&t1);
    int num_batches = ntr / BATCH;

    // ---- training ----------------------------------------------------------
    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        CHECK(cudaMemset(nn.d_loss, 0, sizeof(float)));
        cudaEventRecord(t0);

        for (int b = 0; b < num_batches; b++)
        {
            nn.input = d_xtr + (size_t)b * BATCH * dim;       // this batch (just a pointer offset)
            const int *labels = d_ytr + b * BATCH;
            forward_propagation(nn);
            backward_propagation(nn, labels);
        }

        cudaEventRecord(t1);
        cudaEventSynchronize(t1);                              // the only sync in the epoch
        float ms;  cudaEventElapsedTime(&ms, t0, t1);

        float loss;
        CHECK(cudaMemcpy(&loss, nn.d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        float acc = evaluate(nn, d_xte, h_yte, nte, dim, d_pred);   // not included in the timing
        printf("Epoch %d/%d  loss %.4f  test accuracy %.4f  train time %.1f ms\n",
               epoch + 1, EPOCHS, loss / (num_batches * BATCH), acc, ms);
    }

    nn_free(nn);
    return 0;
}
