
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "neural_network.cuh"
#include "cuda_utils.cuh"
#include "dataset.h"

#define EPOCHS 20
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01f
#define NUM_CLASSES 10
#define SEED 42
#define WARMUP_RUNS 10

namespace fs = std::filesystem;


// Load a batch of images and one-hot targets onto GPU
void LoadBatch(
    Neural_Network* nn,
    const Dataset* dataset,
    const int* indices,
    int count,
    float* d_targets)
{
    int B = nn->batch_size;
    int pixels = nn->layer_sizes[0];

    std::vector<float> inputs(B * pixels, 0.0f);
    std::vector<float> targets(B * NUM_CLASSES, 0.0f);

    for (int i = 0; i < count; i++)
    {
        int index = indices[i];

        std::memcpy(
            inputs.data() + i * pixels,
            dataset->images + (size_t)index * pixels,
            pixels * sizeof(float)
        );

        int label = dataset->labels[index];
        targets[i * NUM_CLASSES + label] = 1.0f;
    }

    CUDA_CHECK(cudaMemcpy(
        nn->outputs[0],
        inputs.data(),
        inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_targets,
        targets.data(),
        targets.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}


// Predict class from output logits
int PredictClass(const float* logits, int classes)
{
    int prediction = 0;

    for (int j = 1; j < classes; j++)
    {
        if (logits[j] > logits[prediction])
            prediction = j;
    }

    return prediction;
}


// Stable cross-entropy from logits
float CrossEntropyFromLogits(
    const float* logits,
    int label,
    int classes)
{
    float max_logit = logits[0];

    for (int j = 1; j < classes; j++)
    {
        max_logit = std::max(max_logit, logits[j]);
    }

    float sum = 0.0f;

    for (int j = 0; j < classes; j++)
    {
        sum += expf(logits[j] - max_logit);
    }

    return logf(sum) + max_logit - logits[label];
}


// Evaluate average loss and classification accuracy
void Evaluate(
    Neural_Network* nn,
    const Dataset* dataset,
    const int* indices,
    int count,
    CUDA_Mode mode,
    float* loss,
    float* accuracy)
{
    int B = nn->batch_size;
    int L = nn->num_layers - 1;

    float* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_targets,
        B * NUM_CLASSES * sizeof(float)
    ));

    std::vector<float> logits(B * NUM_CLASSES);

    double total_loss = 0.0;
    int correct = 0;

    for (int start = 0; start < count; start += B)
    {
        int current = std::min(B, count - start);

        LoadBatch(
            nn, dataset, indices + start,
            current, d_targets
        );

        forward_propagation(nn, mode);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(
            logits.data(),
            nn->outputs[L],
            logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        for (int i = 0; i < current; i++)
        {
            const float* row =
                logits.data() + i * NUM_CLASSES;

            int label =
                dataset->labels[indices[start + i]];

            int prediction =
                PredictClass(row, NUM_CLASSES);

            total_loss += CrossEntropyFromLogits(
                row, label, NUM_CLASSES
            );

            if (prediction == label)
                correct++;
        }
    }

    *loss = static_cast<float>(total_loss / count);
    *accuracy = static_cast<float>(correct) / count;

    CUDA_CHECK(cudaFree(d_targets));
}


// Save predictions for all MNIST test images
void SavePredictions(
    Neural_Network* nn,
    const Dataset* test,
    CUDA_Mode mode,
    const std::string& output_dir)
{
    std::string path =
        output_dir + "/predictions.csv";

    FILE* csv = fopen(path.c_str(), "w");

    if (!csv)
    {
        fprintf(stderr, "Failed to create %s\n",
                path.c_str());
        exit(EXIT_FAILURE);
    }

    fprintf(csv, "index,true_label,predicted_label\n");

    int B = nn->batch_size;
    int L = nn->num_layers - 1;

    std::vector<int> indices(test->count);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<float> logits(B * NUM_CLASSES);

    float* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_targets,
        B * NUM_CLASSES * sizeof(float)
    ));

    int correct = 0;

    for (int start = 0; start < test->count; start += B)
    {
        int current = std::min(
            B, test->count - start
        );

        LoadBatch(
            nn, test,
            indices.data() + start,
            current, d_targets
        );

        forward_propagation(nn, mode);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(
            logits.data(),
            nn->outputs[L],
            logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        for (int i = 0; i < current; i++)
        {
            const float* row =
                logits.data() + i * NUM_CLASSES;

            int index = indices[start + i];
            int prediction = PredictClass(
                row, NUM_CLASSES
            );
            int label = test->labels[index];

            fprintf(csv, "%d,%d,%d\n",
                    index, label, prediction);

            if (prediction == label)
                correct++;
        }
    }

    fclose(csv);

    CUDA_CHECK(cudaFree(d_targets));

    printf("Saved %d predictions to %s\n",
           test->count, path.c_str());

    printf("Prediction Accuracy: %.2f%%\n",
           100.0f * correct / test->count);
}


// Save the first 100 MNIST images as a PGM sheet
void SaveDigitSheet(
    const Dataset* test,
    const std::string& output_dir)
{
    std::string path =
        output_dir + "/test_digits.pgm";

    FILE* pgm = fopen(path.c_str(), "wb");

    if (!pgm)
    {
        fprintf(stderr, "Failed to create %s\n",
                path.c_str());
        exit(EXIT_FAILURE);
    }

    const int grid_cols = 10;
    const int grid_rows = 10;

    int examples = std::min(100, test->count);
    int pixels = test->rows * test->cols;

    int sheet_width = grid_cols * test->cols;
    int sheet_height = grid_rows * test->rows;

    std::vector<unsigned char> sheet(
        sheet_width * sheet_height, 0
    );

    for (int i = 0; i < examples; i++)
    {
        int grid_x = i % grid_cols;
        int grid_y = i / grid_cols;

        for (int y = 0; y < test->rows; y++)
        {
            for (int x = 0; x < test->cols; x++)
            {
                int src =
                    i * pixels + y * test->cols + x;

                int dst =
                    (grid_y * test->rows + y) * sheet_width +
                    grid_x * test->cols + x;

                float value =
                    test->images[src] * 255.0f;

                sheet[dst] =
                    static_cast<unsigned char>(
                        std::lround(value)
                    );
            }
        }
    }

    fprintf(pgm, "P5\n%d %d\n255\n",
            sheet_width, sheet_height);

    fwrite(sheet.data(), 1, sheet.size(), pgm);
    fclose(pgm);

    printf("Saved digit sheet to %s\n",
           path.c_str());
}


// Warm up CUDA before timing actual training
void WarmupGPU(
    Neural_Network* nn,
    const Dataset* dataset,
    const int* indices,
    CUDA_Mode mode,
    float* d_targets)
{
    printf("Warming up GPU (%d runs)...\n",
           WARMUP_RUNS);

    // Upload one full batch
    LoadBatch(
        nn,
        dataset,
        indices,
        nn->batch_size,
        d_targets
    );

    // Warm up forward propagation without updating weights
    for (int i = 0; i < WARMUP_RUNS; i++)
    {
        forward_propagation(nn, mode);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("GPU warm-up complete.\n\n");
}


// Main training program
int main(int argc, char** argv)
{
    // Parse CUDA mode and block size
    if (argc != 3)
    {
        fprintf(stderr,
            "Usage: ./fnn <baseline|optimized> <block_size>\n"
            "Supported block sizes: 8, 16, 32\n"
        );

        return EXIT_FAILURE;
    }

    CUDA_Mode mode;

    if (strcmp(argv[1], "baseline") == 0)
        mode = CUDA_BASELINE;
    else if (strcmp(argv[1], "optimized") == 0)
        mode = CUDA_OPTIMIZED;
    else
    {
        fprintf(stderr, "Invalid CUDA mode\n");
        return EXIT_FAILURE;
    }

    char* end = nullptr;
    long requested = strtol(argv[2], &end, 10);

    if (*argv[2] == '\0' || *end != '\0' ||
        (requested != 8 &&
         requested != 16 &&
         requested != 32))
    {
        fprintf(stderr,
            "Block size must be 8, 16, or 32\n");
        return EXIT_FAILURE;
    }

    int block_size = static_cast<int>(requested);

    const char* mode_name =
        mode == CUDA_OPTIMIZED ? "optimized" : "baseline";

    // Experiment-specific output directory
    std::string output_dir =
        "outputs/" + std::string(mode_name) +
        "/block_" + std::to_string(block_size);

    fs::create_directories(output_dir);

    printf("CUDA Mode: %s\n", mode_name);
    printf("Block Size: %d x %d\n",
           block_size, block_size);
    printf("Output Directory: %s\n\n",
           output_dir.c_str());

    // Load MNIST
    Dataset train{};
    Dataset test{};

    if (dataset_load_mnist(
        &train,
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte") != 0)
    {
        return EXIT_FAILURE;
    }

    if (dataset_load_mnist(
        &test,
        "data/mnist/t10k-images-idx3-ubyte",
        "data/mnist/t10k-labels-idx1-ubyte") != 0)
    {
        dataset_free(&train);
        return EXIT_FAILURE;
    }

    if (train.rows * train.cols != 784 ||
        test.rows * test.cols != 784)
    {
        fprintf(stderr, "Expected 28x28 MNIST images\n");

        dataset_free(&train);
        dataset_free(&test);

        return EXIT_FAILURE;
    }

    // Split into training and validation sets
    int* indices = nullptr;
    int train_count = 0;

    if (dataset_split_indices(
        train.count, 0.8f, SEED,
        &indices, &train_count) != 0)
    {
        fprintf(stderr, "Dataset split failed\n");

        dataset_free(&train);
        dataset_free(&test);

        return EXIT_FAILURE;
    }

    int val_count = train.count - train_count;

    printf("Training Samples:   %d\n", train_count);
    printf("Validation Samples: %d\n", val_count);
    printf("Test Samples:       %d\n\n", test.count);

    // Initialize neural network
    int layers[] = {784, 128, 64, 10};

    Neural_Network nn{};

    nn.num_layers = 4;
    nn.layer_sizes = layers;
    nn.batch_size = BATCH_SIZE;
    nn.learning_rate = LEARNING_RATE;
    nn.block_size = block_size;

    srand(SEED);
    Network(&nn);

    float* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_targets,
        BATCH_SIZE * NUM_CLASSES * sizeof(float)
    ));

    // Warm-up before starting training measurements
    WarmupGPU(
        &nn, &train,
        indices, mode, d_targets
    );

    // Create metrics file
    std::string metrics_path =
        output_dir + "/metrics.csv";

    FILE* metrics = fopen(
        metrics_path.c_str(), "w"
    );

    if (!metrics)
    {
        fprintf(stderr, "Failed to create metrics.csv\n");
        CUDA_CHECK(cudaFree(d_targets));
        FreeNetwork(&nn);
        free(indices);
        dataset_free(&train);
        dataset_free(&test);
        return EXIT_FAILURE;
    }

    fprintf(
        metrics,
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy,"
        "gpu_training_ms,epoch_ms\n"
    );

    // Create CUDA timing events
    cudaEvent_t start_event, stop_event;

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));

    unsigned rng_state = SEED;

    double total_training_ms = 0.0;
    double total_gpu_timeline_ms = 0.0;

    // Training loop
    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {
        // Shuffle training indices
        dataset_shuffle_indices(
            indices, train_count, &rng_state
        );

        // Begin wall-clock timing
        auto cpu_start =
            std::chrono::steady_clock::now();

        // Begin GPU timeline measurement
        CUDA_CHECK(cudaEventRecord(start_event));

        // Mini-batch SGD
        for (int start = 0;
             start + BATCH_SIZE <= train_count;
             start += BATCH_SIZE)
        {
            LoadBatch(
                &nn,
                &train,
                indices + start,
                BATCH_SIZE,
                d_targets
            );

            forward_propagation(&nn, mode);

            backward_propagation(
                &nn, d_targets, mode
            );
        }

        // Stop GPU timeline measurement
        CUDA_CHECK(cudaEventRecord(stop_event));

        // Synchronize ONCE per epoch
        CUDA_CHECK(cudaEventSynchronize(stop_event));

        float gpu_training_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &gpu_training_ms,
            start_event,
            stop_event
        ));

        // Stop wall-clock measurement
        auto cpu_stop =
            std::chrono::steady_clock::now();

        double epoch_ms =
            std::chrono::duration<double, std::milli>(
                cpu_stop - cpu_start
            ).count();

        total_training_ms += epoch_ms;
        total_gpu_timeline_ms += gpu_training_ms;

        // Evaluate model after training
        float train_loss = 0.0f;
        float train_accuracy = 0.0f;
        float val_loss = 0.0f;
        float val_accuracy = 0.0f;

        Evaluate(
            &nn, &train,
            indices, train_count,
            mode,
            &train_loss, &train_accuracy
        );

        Evaluate(
            &nn, &train,
            indices + train_count, val_count,
            mode,
            &val_loss, &val_accuracy
        );

        // Print training results
        printf(
            "Epoch %2d/%d | "
            "Train Loss: %.4f | Train Acc: %.2f%% | "
            "Val Loss: %.4f | Val Acc: %.2f%% | "
            "Train: %.2f ms | GPU Timeline: %.2f ms\n",
            epoch,
            EPOCHS,
            train_loss,
            train_accuracy * 100.0f,
            val_loss,
            val_accuracy * 100.0f,
            epoch_ms,
            gpu_training_ms
        );

        // Save epoch metrics
        fprintf(
            metrics,
            "%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f\n",
            epoch,
            train_loss,
            val_loss,
            train_accuracy,
            val_accuracy,
            gpu_training_ms,
            epoch_ms
        );

        fflush(metrics);
    }

    // Destroy CUDA timing events
    CUDA_CHECK(cudaEventDestroy(start_event));
    CUDA_CHECK(cudaEventDestroy(stop_event));

    fclose(metrics);

    // Print aggregate training performance
    printf("\n=== Training Performance ===\n");

    printf("Total Training Time: %.3f seconds\n",
           total_training_ms / 1000.0);

    printf("Average Epoch Time: %.3f seconds\n",
           total_training_ms / EPOCHS / 1000.0);

    printf("Total GPU Timeline: %.3f seconds\n",
           total_gpu_timeline_ms / 1000.0);

    // Final test evaluation
    std::vector<int> test_indices(test.count);
    std::iota(test_indices.begin(), test_indices.end(), 0);

    float test_loss = 0.0f;
    float test_accuracy = 0.0f;

    Evaluate(
        &nn,
        &test,
        test_indices.data(),
        test.count,
        mode,
        &test_loss,
        &test_accuracy
    );

    printf("\nTest Loss: %.4f\n", test_loss);
    printf("Test Accuracy: %.2f%%\n",
           test_accuracy * 100.0f);

    // Save predictions and digit sheet
    SavePredictions(&nn, &test, mode, output_dir);
    SaveDigitSheet(&test, output_dir);

    // Cleanup
    CUDA_CHECK(cudaFree(d_targets));
    FreeNetwork(&nn);

    free(indices);
    dataset_free(&train);
    dataset_free(&test);

    printf("\nTraining complete.\n");
    printf("Results saved to: %s\n",
           output_dir.c_str());

    return EXIT_SUCCESS;
}
