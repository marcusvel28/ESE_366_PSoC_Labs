
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sys/stat.h>
#include <string>
#include <filesystem>

#include "neural_network.cuh"
#include "cuda_utils.cuh"
#include "dataset.h"

#define EPOCHS 20
#define BATCH_SIZE 32
#define LEARNING_RATE 0.01f
#define NUM_CLASSES 10
#define SEED 42

// Generate test-specific output directory
std::string output_dir =
    "outputs/" + std::string(argv[1]) +
    "/block_" + std::to_string(block_size);

std::filesystem::create_directories(output_dir);

printf("Output Directory: %s\n", output_dir.c_str());

// Load a batch of images and one-hot labels onto the GPU
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
        nn->outputs[0], inputs.data(),
        inputs.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_targets, targets.data(),
        targets.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
}


// Evaluate loss and accuracy from output logits
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
        &d_targets, B * NUM_CLASSES * sizeof(float)
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

            int prediction = 0;
            float max_logit = row[0];

            for (int j = 1; j < NUM_CLASSES; j++)
            {
                if (row[j] > max_logit)
                {
                    max_logit = row[j];
                    prediction = j;
                }
            }

            float sum = 0.0f;

            for (int j = 0; j < NUM_CLASSES; j++)
                sum += expf(row[j] - max_logit);

            int label = dataset->labels[indices[start + i]];

            total_loss +=
                logf(sum) + max_logit - row[label];

            if (prediction == label)
                correct++;
        }
    }

    *loss = (float)(total_loss / count);
    *accuracy = (float)correct / count;

    CUDA_CHECK(cudaFree(d_targets));
}


// Save test predictions and a 10x10 MNIST digit sheet
void SavePredictions(
    Neural_Network* nn,
    const Dataset* test,
    CUDA_Mode mode,
    const std::string& output_dir)
{
    std::string predictions_path =
    output_dir + "/predictions.csv";

    std::string digits_path =
        output_dir + "/test_digits.pgm";

    FILE* csv = fopen(predictions_path.c_str(), "w");
    FILE* pgm = fopen(digits_path.c_str(), "wb");

    if (!csv || !pgm)
    {
        fprintf(stderr, "Failed to create prediction files\n");
        exit(EXIT_FAILURE);
    }

    fprintf(csv, "index,true_label,predicted_label\n");

    const int B = nn->batch_size;
    const int pixels = test->rows * test->cols;
    const int examples = std::min(100, test->count);

    std::vector<int> indices(examples);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<float> logits(B * NUM_CLASSES);

    float* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_targets, B * NUM_CLASSES * sizeof(float)
    ));

    for (int start = 0; start < examples; start += B)
    {
        int current = std::min(B, examples - start);

        LoadBatch(
            nn, test, indices.data() + start,
            current, d_targets
        );

        forward_propagation(nn, mode);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(
            logits.data(), nn->outputs[nn->num_layers - 1],
            logits.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        for (int i = 0; i < current; i++)
        {
            const float* row =
                logits.data() + i * NUM_CLASSES;

            int prediction = 0;

            for (int j = 1; j < NUM_CLASSES; j++)
            {
                if (row[j] > row[prediction])
                    prediction = j;
            }

            int index = indices[start + i];

            fprintf(csv, "%d,%d,%d\n",
                    index, test->labels[index], prediction);
        }
    }

    // Create the 10-column contact sheet expected by plot.py
    int sheet_width = 10 * test->cols;
    int sheet_height = 10 * test->rows;

    std::vector<unsigned char> sheet(
        sheet_width * sheet_height, 0
    );

    for (int i = 0; i < examples; i++)
    {
        int grid_x = i % 10;
        int grid_y = i / 10;

        for (int y = 0; y < test->rows; y++)
        {
            for (int x = 0; x < test->cols; x++)
            {
                int src = i * pixels + y * test->cols + x;

                int dst =
                    (grid_y * test->rows + y) * sheet_width +
                    grid_x * test->cols + x;

                float value = test->images[src] * 255.0f;
                sheet[dst] = (unsigned char)std::lround(value);
            }
        }
    }

    fprintf(pgm, "P5\n%d %d\n255\n",
            sheet_width, sheet_height);

    fwrite(sheet.data(), 1, sheet.size(), pgm);

    fclose(csv);
    fclose(pgm);

    CUDA_CHECK(cudaFree(d_targets));
}



int main(int argc, char** argv)
{
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
    {
        mode = CUDA_BASELINE;
    }
    else if (strcmp(argv[1], "optimized") == 0)
    {
        mode = CUDA_OPTIMIZED;
    }
    else
    {
        fprintf(stderr, "Invalid CUDA mode\n");
        return EXIT_FAILURE;
    }

    char* end = nullptr;
    long requested = strtol(argv[2], &end, 10);

    if (*argv[2] == '\0' || *end != '\0' ||
        (requested != 8 && requested != 16 && requested != 32))
    {
        fprintf(stderr, "Block size must be 8, 16, or 32\n");
        return EXIT_FAILURE;
    }

    int block_size = static_cast<int>(requested);

    printf("CUDA Mode: %s\n", argv[1]);
    printf("Block Size: %d x %d\n",
           block_size, block_size);

    // Keep the rest of your existing main() here


    printf("CUDA FNN - %s\n", argv[1]);

    // Load MNIST
    Dataset train{};
    Dataset test{};

    if (dataset_load_mnist(
        &train,
        "data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte") != 0)
        return EXIT_FAILURE;

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

    // 80% training, 20% validation
    int* indices = nullptr;
    int train_count = 0;

    if (dataset_split_indices(
        train.count, 0.8f, SEED,
        &indices, &train_count) != 0)
    {
        fprintf(stderr, "Failed to split dataset\n");
        dataset_free(&train);
        dataset_free(&test);
        return EXIT_FAILURE;
    }

    int val_count = train.count - train_count;

    printf("Training samples:   %d\n", train_count);
    printf("Validation samples: %d\n", val_count);
    printf("Test samples:       %d\n", test.count);

    // Initialize network
    int layers[] = {784, 128, 64, 10};

    Neural_Network nn{};

    nn.block_size = block_size;
    nn.num_layers = 4;
    nn.layer_sizes = layers;
    nn.batch_size = BATCH_SIZE;
    nn.learning_rate = LEARNING_RATE;

    srand(SEED);
    Network(&nn);

    float* d_targets = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_targets,
        BATCH_SIZE * NUM_CLASSES * sizeof(float)
    ));

    std::string metrics_path =
    output_dir + "/metrics.csv";

    FILE* metrics = fopen(metrics_path.c_str(), "w");

    if (!metrics)
    {
        fprintf(stderr, "Failed to create metrics.csv\n");
        exit(EXIT_FAILURE);
    }

    fprintf(metrics,
        "epoch,train_loss,val_loss,train_accuracy,val_accuracy\n");

    // Shuffle only the training subset
    unsigned rng_state = SEED;

    for (int epoch = 1; epoch <= EPOCHS; epoch++)
    {
        dataset_shuffle_indices(
            indices, train_count, &rng_state
        );

        // Only complete batches during training
        for (int start = 0;
             start + BATCH_SIZE <= train_count;
             start += BATCH_SIZE)
        {
            LoadBatch(
                &nn, &train,
                indices + start,
                BATCH_SIZE,
                d_targets
            );

            forward_propagation(&nn, mode);

            backward_propagation(
                &nn, d_targets, mode
            );
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float train_loss, train_accuracy;
        float val_loss, val_accuracy;

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

        printf(
            "Epoch %2d/%d | "
            "Train Loss: %.4f | Train Acc: %.2f%% | "
            "Val Loss: %.4f | Val Acc: %.2f%%\n",
            epoch, EPOCHS,
            train_loss, 100.0f * train_accuracy,
            val_loss, 100.0f * val_accuracy
        );

        fprintf(metrics, "%d,%.6f,%.6f,%.6f,%.6f\n",
                epoch, train_loss, val_loss,
                train_accuracy, val_accuracy);

        fflush(metrics);
    }

    fclose(metrics);

    // Final evaluation on held-out test dataset
    std::vector<int> test_indices(test.count);
    std::iota(test_indices.begin(), test_indices.end(), 0);

    float test_loss, test_accuracy;

    Evaluate(
        &nn, &test,
        test_indices.data(), test.count,
        mode,
        &test_loss, &test_accuracy
    );

    printf("\nTest Loss: %.4f\n", test_loss);
    printf("Test Accuracy: %.2f%%\n",
           test_accuracy * 100.0f);

    SavePredictions(&nn, &test, mode, output_dir);

    // Cleanup
    CUDA_CHECK(cudaFree(d_targets));

    FreeNetwork(&nn);

    free(indices);
    dataset_free(&train);
    dataset_free(&test);

    printf("\nTraining complete.\n");

    return EXIT_SUCCESS;
}
