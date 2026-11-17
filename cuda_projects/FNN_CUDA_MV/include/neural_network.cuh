#ifndef NEURAL_NETWORK_CUH
#define NEURAL_NETWORK_CUH

#include <cuda_runtime.h>
#include "matrix_operations.cuh"
#include "activation.cuh"
#include "loss.cuh"

// Neural Network structure
struct Neural_Network
{
    int num_layers;
    int* layer_sizes;
    int batch_size;
    float learning_rate;
    int block_size;

    float** weights;
    float** bias;
    float** outputs;
    float** deltas;
    float** weight_gradient;
    float** bias_gradient;

    // Reusable GPU buffers for backpropagation
    float* probabilities;
    float* transposed_weights;
    float* transposed_outputs;
};

// Neural Network initialization and cleanup
void Network(Neural_Network* nn);
void FreeNetwork(Neural_Network* nn);

// Neural Network functions
void forward_propagation(Neural_Network* nn, CUDA_Mode mode);
void backward_propagation(Neural_Network* nn, const float* targets, CUDA_Mode mode);

// Bias gradient: sum deltas across the batch
__global__ void BiasGradient(
    const float* delta, float* gradient,
    int rows, int cols
);

// SGD parameter update
__global__ void SGDUpdate(
    float* parameters, const float* gradients,
    float learning_rate, int batch_size, int n
);


#endif
