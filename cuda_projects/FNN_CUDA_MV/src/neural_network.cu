#include "neural_network.cuh"
#include "cuda_utils.cuh"
#include <cstdlib>
#include <cmath>

// Allocate GPU memory and initialize network
void Network(Neural_Network* nn)
{
    int L = nn->num_layers - 1;
    int B = nn->batch_size;

    // Allocate host pointer arrays
    nn->weights = new float*[L]();
    nn->bias = new float*[L]();
    nn->outputs = new float*[L + 1]();
    nn->deltas = new float*[L]();
    nn->weight_gradient = new float*[L]();
    nn->bias_gradient = new float*[L]();

    // Allocate input activation buffer
    CUDA_CHECK(cudaMalloc(&nn->outputs[0],
               B * nn->layer_sizes[0] * sizeof(float)));

    // Allocate each layer's matrices
    for (int i = 0; i < L; i++)
    {
        int K = nn->layer_sizes[i];
        int N = nn->layer_sizes[i + 1];

        CUDA_CHECK(cudaMalloc(&nn->weights[i], K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->bias[i], N * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&nn->outputs[i + 1], B * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->deltas[i], B * N * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&nn->weight_gradient[i], K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&nn->bias_gradient[i], N * sizeof(float)));

        // Initialize biases to zero
        CUDA_CHECK(cudaMemset(nn->bias[i], 0, N * sizeof(float)));

        // Temporary CPU weight buffer
        float* h_weights = new float[K * N];

        // He initialization for ReLU layers
        float scale = sqrtf(6.0f / K);

        for (int j = 0; j < K * N; j++)
        {
            // Uniform approximation for initial weights
            float r = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
            h_weights[j] = r * scale;
        }

        CUDA_CHECK(cudaMemcpy(nn->weights[i], h_weights,
                   K * N * sizeof(float),
                   cudaMemcpyHostToDevice));

        delete[] h_weights;
    }
}

// Forward propagation through all layers

void forward_propagation(Neural_Network* nn, CUDA_Mode mode)
{
    for (int i = 0; i < nn->num_layers - 1; i++)
    {
        // Matrix dimensions
        int M = nn->batch_size;
        int N = nn->layer_sizes[i + 1];
        int K = nn->layer_sizes[i];

        // 2D CUDA grid for matrix operations
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Z = A * W
        if (mode == CUDA_OPTIMIZED)
        {
            MatrixMultiplication_Optimized<<<grid, block>>>(
                nn->outputs[i],
                nn->weights[i],
                nn->outputs[i + 1],
                M, N, K
            );
        }
        else
        {
            MatrixMultiplication_Baseline<<<grid, block>>>(
                nn->outputs[i],
                nn->weights[i],
                nn->outputs[i + 1],
                M, N, K
            );
        }

        // Z = Z + b
        AddBias<<<grid, block>>>(
            nn->outputs[i + 1],
            nn->bias[i],
            M, N
        );

        // Apply ReLU to hidden layers only
        if (i < nn->num_layers - 2)
        {
            int n = M * N;

            int threads = 256;
            int blocks = (n + threads - 1) / threads;

            relu<<<blocks, threads>>>(
                nn->outputs[i + 1],
                nn->outputs[i + 1],
                n
            );
        }
    }
}


void backward_propagation(
    Neural_Network* nn, const float* targets,
    CUDA_Mode mode)
{
    int L = nn->num_layers - 1;
    int B = nn->batch_size;
    int output_size = nn->layer_sizes[L];

    int threads = 256;

    // Find maximum temporary buffer sizes
    int max_weights = 0;
    int max_activation = 0;

    for (int i = 0; i < L; i++)
    {
        int K = nn->layer_sizes[i];
        int N = nn->layer_sizes[i + 1];

        if (K * N > max_weights)
            max_weights = K * N;

        if (B * K > max_activation)
            max_activation = B * K;
    }

    // Temporary GPU buffers
    float* probabilities = nullptr;
    float* transposed_weights = nullptr;
    float* transposed_outputs = nullptr;

    CUDA_CHECK(cudaMalloc(
        &probabilities, B * output_size * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &transposed_weights, max_weights * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &transposed_outputs, max_activation * sizeof(float)));

    // Compute softmax probabilities
    int blocks = (B + threads - 1) / threads;

    softmax<<<blocks, threads>>>(
        nn->outputs[L], probabilities, B, output_size
    );

    // Output layer delta
    int n = B * output_size;
    blocks = (n + threads - 1) / threads;

    output_delta<<<blocks, threads>>>(
        probabilities, targets,
        nn->deltas[L - 1], n
    );

    // Calculate gradients from output to input
    for (int i = L - 1; i >= 0; i--)
    {
        int K = nn->layer_sizes[i];
        int N = nn->layer_sizes[i + 1];

        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        // Calculate hidden layer delta
        if (i < L - 1)
        {
            int nextN = nn->layer_sizes[i + 2];

            // Transpose next layer weights
            dim3 gridT(
                (nextN + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            MatrixTranspose<<<gridT, block>>>(
                nn->weights[i + 1],
                transposed_weights,
                N, nextN
            );

            // delta[i] = delta[i+1] * W[i+1]^T
            dim3 gridM(
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (B + BLOCK_SIZE - 1) / BLOCK_SIZE
            );

            if (mode == CUDA_OPTIMIZED)
            {
                MatrixMultiplication_Optimized<<<gridM, block>>>(
                    nn->deltas[i + 1],
                    transposed_weights,
                    nn->deltas[i],
                    B, N, nextN
                );
            }
            else
            {
                MatrixMultiplication_Baseline<<<gridM, block>>>(
                    nn->deltas[i + 1],
                    transposed_weights,
                    nn->deltas[i],
                    B, N, nextN
                );
            }

            // Apply ReLU derivative
            int count = B * N;
            int grid = (count + threads - 1) / threads;

            relu_backward<<<grid, threads>>>(
                nn->deltas[i],
                nn->outputs[i + 1],
                count
            );
        }

        // Transpose activations: A^T
        dim3 gridT(
            (K + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (B + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        MatrixTranspose<<<gridT, block>>>(
            nn->outputs[i],
            transposed_outputs,
            B, K
        );

        // Weight gradient = A^T * delta
        dim3 gridW(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (K + BLOCK_SIZE - 1) / BLOCK_SIZE
        );

        if (mode == CUDA_OPTIMIZED)
        {
            MatrixMultiplication_Optimized<<<gridW, block>>>(
                transposed_outputs,
                nn->deltas[i],
                nn->weight_gradient[i],
                K, N, B
            );
        }
        else
        {
            MatrixMultiplication_Baseline<<<gridW, block>>>(
                transposed_outputs,
                nn->deltas[i],
                nn->weight_gradient[i],
                K, N, B
            );
        }

        // Bias gradient
        blocks = (N + threads - 1) / threads;

        BiasGradient<<<blocks, threads>>>(
            nn->deltas[i],
            nn->bias_gradient[i],
            B, N
        );
    }

    // Update weights and biases after all gradients
    for (int i = 0; i < L; i++)
    {
        int K = nn->layer_sizes[i];
        int N = nn->layer_sizes[i + 1];

        int weight_count = K * N;
        int weight_blocks =
            (weight_count + threads - 1) / threads;

        SGDUpdate<<<weight_blocks, threads>>>(
            nn->weights[i],
            nn->weight_gradient[i],
            nn->learning_rate,
            B,
            weight_count
        );

        int bias_blocks = (N + threads - 1) / threads;

        SGDUpdate<<<bias_blocks, threads>>>(
            nn->bias[i],
            nn->bias_gradient[i],
            nn->learning_rate,
            B,
            N
        );
    }

    CUDA_CHECK(cudaGetLastError());

    // Release temporary buffers
    CUDA_CHECK(cudaFree(probabilities));
    CUDA_CHECK(cudaFree(transposed_weights));
    CUDA_CHECK(cudaFree(transposed_outputs));
}


void FreeNetwork(Neural_Network* nn)
{
    int L = nn->num_layers - 1;

    // Free input activation buffer
    CUDA_CHECK(cudaFree(nn->outputs[0]));

    // Free all GPU matrices
    for (int i = 0; i < L; i++)
    {
        CUDA_CHECK(cudaFree(nn->weights[i]));
        CUDA_CHECK(cudaFree(nn->bias[i]));
        CUDA_CHECK(cudaFree(nn->outputs[i + 1]));
        CUDA_CHECK(cudaFree(nn->deltas[i]));
        CUDA_CHECK(cudaFree(nn->weight_gradient[i]));
        CUDA_CHECK(cudaFree(nn->bias_gradient[i]));
    }

    // Free host pointer arrays
    delete[] nn->weights;
    delete[] nn->bias;
    delete[] nn->outputs;
    delete[] nn->deltas;
    delete[] nn->weight_gradient;
    delete[] nn->bias_gradient;

    // Reset pointers
    nn->weights = nullptr;
    nn->bias = nullptr;
    nn->outputs = nullptr;
    nn->deltas = nullptr;
    nn->weight_gradient = nullptr;
    nn->bias_gradient = nullptr;
}


 // Bias gradient: sum deltas across the batch
__global__ void BiasGradient(
    const float* delta, float* gradient,
    int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols)
    {
        float sum = 0.0f;

        for (int row = 0; row < rows; row++)
        {
            sum += delta[row * cols + col];
        }

        gradient[col] = sum;
    }
}


// SGD parameter update
__global__ void SGDUpdate(
    float* parameters, const float* gradients,
    float learning_rate, int batch_size, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float scale = learning_rate / batch_size;

        parameters[idx] -= scale * gradients[idx];
    }
}
