#include "loss.cuh"
#include <cmath>

// Softmax cross-entropy derivative
__global__ void output_delta(
    const float* probabilities,
    const float* targets,
    float* delta,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        delta[idx] = probabilities[idx] - targets[idx];
    }
}


// Cross-entropy loss for each sample
__global__ void cross_entropy(
    const float* probabilities,
    const float* targets,
    float* losses,
    int rows,
    int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows)
    {
        float loss = 0.0f;

        for (int j = 0; j < cols; j++)
        {
            int idx = row * cols + j;

            float p = fmaxf(probabilities[idx], 1e-7f);

            loss -= targets[idx] * logf(p);
        }

        losses[row] = loss;
    }
}
