//**************************************************************************
//
// File Name : "neural_network.cuh"
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
#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include "matrix_operations.cuh"

// Convention: layer i reads outputs[i-1] (or `input` for layer 0) and writes outputs[i].
// layer_sizes = {784, HIDDEN, 10}, so there are num_layers-1 weight layers.
typedef struct {
    int    num_layers;
    int   *layer_sizes;     // host array
    int    batch_size;
    float  lr;
    const float *input;     // device pointer to the current batch (set before each step)

    float **weights;        // weights[i]: layer_sizes[i]   x layer_sizes[i+1]
    float **bias;           // bias[i]   : layer_sizes[i+1]
    float **outputs;        // outputs[i]: batch x layer_sizes[i+1]   (post-ReLU; logits for last)
    float **deltas;         // deltas[i] : batch x layer_sizes[i+1]
    float **dW;             // dW[i]     : layer_sizes[i]   x layer_sizes[i+1]   (stored gradient)
    float **db;             // db[i]     : layer_sizes[i+1]                      (stored gradient)
    float **inT;            // inT[i]    : layer_sizes[i]   x batch   (scratch: input^T)
    float **WT;             // WT[i]     : layer_sizes[i+1] x layer_sizes[i]   (scratch: W^T)
    float  *d_loss;         // one float on the GPU; accumulates the summed loss
} Neural_Network;

Neural_Network nn_create(const int *sizes, int num_layers, int batch, float lr);
void nn_free(Neural_Network nn);

// Forward pass. Set nn.input to the batch first.
void forward_propagation(Neural_Network nn);

// Backward pass + weight update. Run forward_propagation on the SAME batch first.
// labels: device array of nn.batch_size ints (0-9).
void backward_propagation(Neural_Network nn, const int *labels);

// Fraction of the n images (already on the GPU in d_x) that the network classifies correctly.
// n must be a multiple of nn.batch_size. d_pred is a device scratch array of batch_size ints.
float evaluate(Neural_Network nn, const float *d_x, const int *h_labels, int n, int dim, int *d_pred);

#endif // NEURAL_NETWORK_CUH
