#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#include "neural_network.cuh"
#include "cuda_utils.cuh"

int main()
{
    int layers[] = {784, 128, 64, 10};

    Neural_Network nn{};

    nn.num_layers = 4;
    nn.layer_sizes = layers;
    nn.batch_size = 32;
    nn.learning_rate = 0.01f;

    const int B = nn.batch_size;
    const int input_size = layers[0];
    const int output_size = layers[3];

    // Deterministic weight initialization
    srand(42);

    printf("Initializing network...\n");
    Network(&nn);

    // Generate synthetic input data
    std::vector<float> h_inputs(B * input_size);

    for (int i = 0; i < B * input_size; i++)
    {
        h_inputs[i] = (i % 256) / 255.0f;
    }

    CUDA_CHECK(cudaMemcpy(
        nn.outputs[0],
        h_inputs.data(),
        h_inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Generate one-hot target labels
    std::vector<float> h_targets(B * output_size, 0.0f);

    for (int i = 0; i < B; i++)
    {
        h_targets[i * output_size + (i % output_size)] = 1.0f;
    }

    float* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_targets,
        h_targets.size() * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_targets,
        h_targets.data(),
        h_targets.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Store results from both forward passes
    std::vector<float> baseline_outputs(B * output_size);
    std::vector<float> optimized_outputs(B * output_size);

    // Baseline forward propagation
    printf("Running baseline CUDA...\n");

    forward_propagation(&nn, CUDA_BASELINE);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        baseline_outputs.data(),
        nn.outputs[3],
        baseline_outputs.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Optimized forward propagation
    printf("Running optimized CUDA...\n");

    forward_propagation(&nn, CUDA_OPTIMIZED);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        optimized_outputs.data(),
        nn.outputs[3],
        optimized_outputs.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Compare forward propagation outputs
    float max_error = 0.0f;

    for (size_t i = 0; i < baseline_outputs.size(); i++)
    {
        float error = fabsf(
            baseline_outputs[i] - optimized_outputs[i]
        );

        max_error = std::max(max_error, error);
    }

    printf("Maximum output difference: %.8f\n", max_error);

    if (std::isfinite(max_error) && max_error < 1e-3f)
    {
        printf("Forward propagation comparison PASSED\n");
    }
    else
    {
        printf("Forward propagation comparison FAILED\n");
        CUDA_CHECK(cudaFree(d_targets));
        FreeNetwork(&nn);
        return EXIT_FAILURE;
    }

    // Save weights before backpropagation
    const int weight_count = layers[0] * layers[1];

    std::vector<float> weights_before(weight_count);
    std::vector<float> weights_after(weight_count);

    CUDA_CHECK(cudaMemcpy(
        weights_before.data(),
        nn.weights[0],
        weight_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Run one backpropagation step
    printf("Running backpropagation...\n");

    backward_propagation(&nn, d_targets, CUDA_OPTIMIZED);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        weights_after.data(),
        nn.weights[0],
        weight_count * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    // Check whether weights changed
    float max_weight_change = 0.0f;

    for (int i = 0; i < weight_count; i++)
    {
        float change = fabsf(
            weights_after[i] - weights_before[i]
        );

        max_weight_change =
            std::max(max_weight_change, change);
    }

    printf("Maximum weight change: %.8f\n", max_weight_change);

    if (std::isfinite(max_weight_change) &&
        max_weight_change > 0.0f)
    {
        printf("SGD weight update comparison PASSED\n");
    }
    else
    {
        printf("SGD weight update comparison FAILED\n");
        CUDA_CHECK(cudaFree(d_targets));
        FreeNetwork(&nn);
        return EXIT_FAILURE;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_targets));
    FreeNetwork(&nn);

    printf("GPU memory freed successfully!\n");

    return EXIT_SUCCESS;
}
