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
#define BATCH   100        // divides both 60000 and 10000, so every batch is full
#define HIDDEN  64
#define EPOCHS  5
#define LR      0.1f
#define DATA_DIR "data/MNIST/raw"

// ---------------------------------------------------------------------------
// MNIST loading (IDX files -> flat float / int arrays on the host)
// ---------------------------------------------------------------------------
static unsigned int read_be32(FILE *f)
{
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "bad MNIST file\n"); exit(1); }
    return ((unsigned)b[0] << 24) | ((unsigned)b[1] << 16) | ((unsigned)b[2] << 8) | b[3];
}

static float* load_images(const char *name, int *n, int *dim)
{
    char path[512];  snprintf(path, sizeof(path), "%s/%s", DATA_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    read_be32(f);                                   // magic number
    *n = read_be32(f);
    int rows = read_be32(f), cols = read_be32(f);
    *dim = rows * cols;
    size_t total = (size_t)(*n) * (*dim);
    unsigned char *raw = (unsigned char*)malloc(total);
    if (fread(raw, 1, total, f) != total) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    float *x = (float*)malloc(total * sizeof(float));
    for (size_t i = 0; i < total; i++) x[i] = raw[i] / 255.0f;
    free(raw);
    return x;
}

static int* load_labels(const char *name, int *n)
{
    char path[512];  snprintf(path, sizeof(path), "%s/%s", DATA_DIR, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    read_be32(f);
    *n = read_be32(f);
    unsigned char *raw = (unsigned char*)malloc(*n);
    if (fread(raw, 1, *n, f) != (size_t)*n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    int *y = (int*)malloc(*n * sizeof(int));
    for (int i = 0; i < *n; i++) y[i] = raw[i];
    free(raw);
    return y;
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

    // ---- load MNIST, then put the whole dataset on the GPU once ------------
    int ntr, nte, dim, dim2, n2;
    float *h_xtr = load_images("train-images-idx3-ubyte", &ntr, &dim);
    int   *h_ytr = load_labels("train-labels-idx1-ubyte", &n2);
    float *h_xte = load_images("t10k-images-idx3-ubyte", &nte, &dim2);
    int   *h_yte = load_labels("t10k-labels-idx1-ubyte", &n2);

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
