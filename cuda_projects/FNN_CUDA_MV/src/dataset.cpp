
#include "dataset.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <cstddef>

// Read a 32-bit unsigned integer in big-endian format
static uint32_t read_u32be(FILE* f, int* ok)
{
    unsigned char b[4];

    if (fread(b, 1, 4, f) != 4)
    {
        *ok = 0;
        return 0;
    }

    return ((uint32_t)b[0] << 24) |
           ((uint32_t)b[1] << 16) |
           ((uint32_t)b[2] << 8)  |
            (uint32_t)b[3];
}


// Free allocated dataset memory
void dataset_free(Dataset* d)
{
    if (!d)
        return;

    free(d->images);
    free(d->labels);

    memset(d, 0, sizeof(*d));
}


// Load MNIST images and labels from IDX files
int dataset_load_mnist(
    Dataset* d,
    const char* images_path,
    const char* labels_path)
{
    if (!d || !images_path || !labels_path)
        return -1;

    memset(d, 0, sizeof(*d));

    FILE* fi = fopen(images_path, "rb");

    if (!fi)
    {
        perror(images_path);
        return -1;
    }

    FILE* fl = fopen(labels_path, "rb");

    if (!fl)
    {
        perror(labels_path);
        fclose(fi);
        return -1;
    }

    int ok = 1;

    // Read IDX image header
    uint32_t magic_i = read_u32be(fi, &ok);
    uint32_t n       = read_u32be(fi, &ok);
    uint32_t rows    = read_u32be(fi, &ok);
    uint32_t cols    = read_u32be(fi, &ok);

    // Read IDX label header
    uint32_t magic_l  = read_u32be(fl, &ok);
    uint32_t n_labels = read_u32be(fl, &ok);

    // Validate dataset headers
    if (!ok ||
        magic_i != 2051 ||
        magic_l != 2049 ||
        n != n_labels ||
        n == 0 ||
        n > INT_MAX ||
        rows == 0 ||
        cols == 0 ||
        rows > 10000 ||
        cols > 10000)
    {
        fprintf(stderr, "Invalid MNIST IDX headers\n");
        goto error;
    }

    {
        size_t pixels = (size_t)rows * cols;

        // Check for integer overflow
        if ((size_t)n > SIZE_MAX / pixels ||
            (size_t)n * pixels > SIZE_MAX / sizeof(float))
        {
            fprintf(stderr, "MNIST size overflow\n");
            goto error;
        }

        size_t total = (size_t)n * pixels;

        // Allocate CPU memory for dataset
        d->images = (float*)malloc(total * sizeof(float));
        d->labels = (unsigned char*)malloc((size_t)n);

        if (!d->images || !d->labels)
            goto error;

        unsigned char* buffer =
            (unsigned char*)malloc(pixels);

        if (!buffer)
            goto error;

        // Load and normalize images
        for (uint32_t k = 0; k < n; k++)
        {
            if (fread(buffer, 1, pixels, fi) != pixels)
            {
                free(buffer);
                goto error;
            }

            for (size_t j = 0; j < pixels; j++)
            {
                d->images[(size_t)k * pixels + j] =
                    buffer[j] / 255.0f;
            }
        }

        free(buffer);
    }

    // Load labels
    if (fread(d->labels, 1, n, fl) != n)
        goto error;

    // Validate digit labels
    for (uint32_t k = 0; k < n; k++)
    {
        if (d->labels[k] > 9)
        {
            fprintf(stderr, "Invalid MNIST label\n");
            goto error;
        }
    }

    // Store dataset dimensions
    d->count = (int)n;
    d->rows  = (int)rows;
    d->cols  = (int)cols;

    fclose(fi);
    fclose(fl);

    return 0;

error:
    fprintf(stderr,
            "Failed to load MNIST files: %s / %s\n",
            images_path, labels_path);

    fclose(fi);
    fclose(fl);

    dataset_free(d);

    return -1;
}


// Simple pseudorandom number generator
static unsigned next_rand(unsigned* state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}


// Fisher-Yates shuffle
void dataset_shuffle_indices(
    int* indices,
    int count,
    unsigned* rng_state)
{
    if (!indices || !rng_state)
        return;

    for (int i = count - 1; i > 0; i--)
    {
        int j = (int)(
            next_rand(rng_state) % (unsigned)(i + 1)
        );

        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}


// Split dataset into training and validation indices
int dataset_split_indices(
    int count,
    float train_ratio,
    unsigned seed,
    int** indices,
    int* train_count)
{
    if (count < 2 ||
        train_ratio <= 0.0f ||
        train_ratio >= 1.0f ||
        !indices ||
        !train_count)
    {
        return -1;
    }

    *indices = (int*)malloc(
        (size_t)count * sizeof(int)
    );

    if (!*indices)
        return -1;

    // Initialize sequential indices
    for (int i = 0; i < count; i++)
    {
        (*indices)[i] = i;
    }

    // Shuffle indices before splitting
    dataset_shuffle_indices(indices[0], count, &seed);

    // Determine training set size
    *train_count = (int)(count * train_ratio);

    if (*train_count < 1)
        *train_count = 1;

    if (*train_count >= count)
        *train_count = count - 1;

    return 0;
}
