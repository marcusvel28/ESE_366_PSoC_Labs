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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "matrix_operations.cuh"

//Neural Network struct
struct Neural_Network{
    int num_layers;
    int* layer_sizes;
    int batch_size;
    float learning_rate;
    float** bias;
    float** weights;
    float** outputs;
    float** deltas;
    float** weight_gradient;
    float** bias_gradient;
}nn;

//Neural Network functions
void forward_propagation(Neural_Network nn);                                //forward propagation function that calls forward kernel for each layer
void backward_propagation(Neural_Network nn, float* t);                     //backward propagation function that calls backward kernel for each layer

//Neural Network kernels
__global__ void forward(float* w, float* b, float* z);						//move forward through the network to compute outputs
__global__ void relu_activate(float* y, float* z);						    //activation function kernel
__global__ void loss(float* e, float* z, float* t);							//loss function kernel
__global__ void backward(float* x, float* t, float *l);				        //backpropagation kernel to update weights and biases

#endif
