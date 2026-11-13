//**************************************************************************
//
// File Name : "neural_network.cu"
// Title : neural_network
// Date : 11/09/2026
// Version : 1.0
// Target Hardware : NVIDIA Jetson Orin Nano
// Author : Adam Roccanova
//
// Revision History : Initial version
//
//
//**************************************************************************

#include <string.h>
#include <math.h>
#include "neural_network.cuh"

static float* dev_zeros(size_t count)
{
    float *p;
    CHECK(cudaMalloc(&p, count * sizeof(float)));
    CHECK(cudaMemset(p, 0, count * sizeof(float)));
    return p;
}

Neural_Network nn_create(const int *sizes, int num_layers, int batch, float lr)
{
    Neural_Network nn;
    int L = num_layers - 1;
    nn.num_layers = num_layers;  nn.batch_size = batch;  nn.lr = lr;  nn.input = NULL;
    nn.layer_sizes = (int*)malloc(num_layers * sizeof(int));
    memcpy(nn.layer_sizes, sizes, num_layers * sizeof(int));

    nn.weights = (float**)malloc(L * sizeof(float*));
    nn.bias    = (float**)malloc(L * sizeof(float*));
    nn.outputs = (float**)malloc(L * sizeof(float*));
    nn.deltas  = (float**)malloc(L * sizeof(float*));
    nn.inT     = (float**)malloc(L * sizeof(float*));
    nn.WT      = (float**)malloc(L * sizeof(float*));

    for (int i = 0; i < L; i++)
    {
        int Kin = sizes[i], Nout = sizes[i + 1];

        // random weights in [-limit, limit] (He initialization, suited to ReLU)
        float *w = (float*)malloc((size_t)Kin * Nout * sizeof(float));
        float limit = sqrtf(6.0f / Kin);
        for (int j = 0; j < Kin * Nout; j++) w[j] = limit * (2.0f * rand() / RAND_MAX - 1.0f);
        CHECK(cudaMalloc(&nn.weights[i], (size_t)Kin * Nout * sizeof(float)));
        CHECK(cudaMemcpy(nn.weights[i], w, (size_t)Kin * Nout * sizeof(float), cudaMemcpyHostToDevice));
        free(w);

        nn.bias[i]    = dev_zeros(Nout);
        nn.outputs[i] = dev_zeros((size_t)batch * Nout);
        nn.deltas[i]  = dev_zeros((size_t)batch * Nout);
        nn.inT[i]     = dev_zeros((size_t)Kin * batch);
        nn.WT[i]      = dev_zeros((size_t)Nout * Kin);
    }
    nn.d_loss = dev_zeros(1);
    return nn;
}

void nn_free(Neural_Network nn)
{
    for (int i = 0; i < nn.num_layers - 1; i++) {
        cudaFree(nn.weights[i]); cudaFree(nn.bias[i]);   cudaFree(nn.outputs[i]);
        cudaFree(nn.deltas[i]);  cudaFree(nn.inT[i]);    cudaFree(nn.WT[i]);
    }
    cudaFree(nn.d_loss);
    free(nn.weights); free(nn.bias); free(nn.outputs);
    free(nn.deltas);  free(nn.inT);  free(nn.WT);  free(nn.layer_sizes);
}

// ---------------------------------------------------------------------------
// Forward pass.  Set nn.input to the batch first.
// ---------------------------------------------------------------------------
void forward_propagation(Neural_Network nn)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int M = nn.batch_size;

    for (int i = 0; i < nn.num_layers - 1; i++)
    {
        int K = nn.layer_sizes[i];                       // inputs to this layer
        int N = nn.layer_sizes[i + 1];                   // neurons in this layer
        const float *in = (i == 0) ? nn.input : nn.outputs[i - 1];
        int relu = (i < nn.num_layers - 2);              // no ReLU on the last layer

        forward<<<grid2D(N, M), block>>>(in, nn.weights[i], nn.bias[i], nn.outputs[i],
                                         M, N, K, relu);
    }
    CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Backward pass + weight update. Run forward_propagation on the SAME batch first.
// labels: device array of `batch_size` ints (0-9).
//
//   PHASE 1: delta for the output layer
//   PHASE 2: delta for each hidden layer, going backward
//   PHASE 3: update every layer's weights and biases
//
// Phase 3 comes after ALL deltas are computed, so phase 2 always sees the old weights.
// ---------------------------------------------------------------------------
void backward_propagation(Neural_Network nn, const int *labels)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    int M = nn.batch_size;
    int L = nn.num_layers - 1;

    // ---- PHASE 1: output layer delta (and the loss) ------------------------
    softmax_delta<<<GRID1D(M), THREADS_1D>>>(nn.outputs[L - 1], labels, nn.deltas[L - 1],
                                             nn.d_loss, M, nn.layer_sizes[L]);

    // ---- PHASE 2: hidden layer deltas, last hidden layer first -------------
    //   delta[i-1] = (delta[i] * W[i]^T) masked by relu'(outputs[i-1])
    for (int i = L - 1; i >= 1; i--)
    {
        int Kin  = nn.layer_sizes[i];
        int Nout = nn.layer_sizes[i + 1];

        // W is Kin x Nout -> WT is Nout x Kin
        matrix_transpose<<<grid2D(Nout, Kin), block>>>(nn.weights[i], nn.WT[i], Kin, Nout);

        // deltas[i-1] (M x Kin) = deltas[i] (M x Nout) * WT (Nout x Kin)
        delta_backward<<<grid2D(Kin, M), block>>>(nn.deltas[i], nn.WT[i], nn.deltas[i - 1],
                                                  nn.outputs[i - 1], M, Kin, Nout);
    }

    // ---- PHASE 3: update weights and biases of every layer -----------------
    //   W -= lr * (input^T * delta)        b -= lr * (sum of delta over the batch)
    for (int i = 0; i < L; i++)
    {
        int Kin  = nn.layer_sizes[i];
        int Nout = nn.layer_sizes[i + 1];
        const float *in = (i == 0) ? nn.input : nn.outputs[i - 1];

        // in is M x Kin -> inT is Kin x M
        matrix_transpose<<<grid2D(Kin, M), block>>>(in, nn.inT[i], M, Kin);

        // W (Kin x Nout) -= lr * inT (Kin x M) * delta (M x Nout)
        weight_grad_update<<<grid2D(Nout, Kin), block>>>(nn.inT[i], nn.deltas[i], nn.weights[i],
                                                         nn.lr, Kin, Nout, M);

        col_sum_update<<<GRID1D(Nout), THREADS_1D>>>(nn.deltas[i], nn.bias[i], nn.lr, M, Nout);
    }
    CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Evaluation: fraction of images whose largest output matches the label
// ---------------------------------------------------------------------------
float evaluate(Neural_Network nn, const float *d_x, const int *h_labels, int n, int dim, int *d_pred)
{
    int M = nn.batch_size;
    int last = nn.num_layers - 2;                        // index of the output layer
    int *pred = (int*)malloc(M * sizeof(int));
    int correct = 0;

    for (int start = 0; start < n; start += M)
    {
        nn.input = d_x + (size_t)start * dim;
        forward_propagation(nn);
        argmax_rows<<<GRID1D(M), THREADS_1D>>>(nn.outputs[last], d_pred, M,
                                               nn.layer_sizes[nn.num_layers - 1]);
        CHECK(cudaMemcpy(pred, d_pred, M * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < M; i++) correct += (pred[i] == h_labels[start + i]);
    }
    free(pred);
    return (float)correct / n;
}
