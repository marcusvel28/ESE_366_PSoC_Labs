
#ifndef DATASET_H
#define DATASET_H

#include <stddef.h>

// MNIST dataset structure
typedef struct
{
    float* images;          // Normalized image pixels [0, 1]
    unsigned char* labels;  // Digit labels [0, 9]
    int count;
    int rows;
    int cols;
} Dataset;

// Load MNIST images and labels from IDX files
int dataset_load_mnist(
    Dataset* d,
    const char* images_path,
    const char* labels_path
);

// Free dataset memory
void dataset_free(Dataset* d);

// Split dataset into training and validation indices
int dataset_split_indices(
    int count,
    float train_ratio,
    unsigned seed,
    int** indices,
    int* train_count
);

// Shuffle dataset indices using Fisher-Yates
void dataset_shuffle_indices(
    int* indices,
    int count,
    unsigned* rng_state
);

#endif
