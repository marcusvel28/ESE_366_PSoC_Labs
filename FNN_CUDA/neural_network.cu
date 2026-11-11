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
#include "neural_network.cuh"
#define BLOCK_SIZE 16

void forward_propagation(Neural_Network nn)
{
    cudaFuncSetCacheConfig(forward, cudaFuncCachePreferL1);

    for (int i = 0; i < nn.num_layers - 1; i++)
    {
        dim3 gridDim((nn.layer_sizes[i] + blockDim.x - 1) / blockDim.x, (nn.batch_size + blockDim.y - 1) / blockDim.y);
        forward<<gridDim, blockDim>>>(nn.weights[i], nn.bias[i], nn.outputs[i]);
        cudaDeviceSynchronize();    //may be optional, might delete later
    }
}   

void backward_propagation(Neural_Network nn, float* t)
{
    cudaFuncSetCacheConfig(forward, cudaFuncCachePreferL1);
}                           

__global__ void forward(float* w, float* b, float* z)
{
    float* y;
    float* wt;
    cudaMalloc(&y, sizeof(z));
    cudaMalloc(&wt, sizeof(w));
    
    /* 
    Call relu_activate kernel
    
    int n = nn.batch_size * nn.layer_sizes[i + 1];

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    relu_activate<<<blocks, threads>>>(y, z, n);
    
    */

    
}	

__global__ void relu_activate(float* y, float* z, int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; // 

    if (idx < n) // RELU function check
    {
        y[idx] = (z[idx] > 0.0f) ? z[idx] : 0.0f;
    }
}

__global__ void loss(float* e, float* z, float* t)
{

}	

__global__ void backward(float* x, float* t, float *l) // activation function derivative
{

}				        