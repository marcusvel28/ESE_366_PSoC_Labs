#include "activation.cuh"
#include <cmath>

// ReLU: y = max(0, z)
__global__ void relu(float* y, const float* z, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        y[idx] = (z[idx] > 0.0f) ? z[idx] : 0.0f;
    }
}


// Apply ReLU derivative to an existing gradient
__global__ void relu_backward(
    float* delta, const float* activation, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        if (activation[idx] <= 0.0f)
        {
            delta[idx] = 0.0f;
        }
    }
}


// Stable softmax: one thread per row
__global__ void softmax(
    const float* logits, float* probabilities,
    int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float max_val = logits[row * cols];

        for (int j = 1; j < cols; j++)
        {
            max_val = fmaxf(
                max_val, logits[row * cols + j]
            );
        }

        float sum = 0.0f;

        for (int j = 0; j < cols; j++)
        {
            float value = expf(
                logits[row * cols + j] - max_val
            );

            probabilities[row * cols + j] = value;
            sum += value;
        }

        for (int j = 0; j < cols; j++)
        {
            probabilities[row * cols + j] /= sum;
        }
    }
}
