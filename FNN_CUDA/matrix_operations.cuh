//**************************************************************************
//
// File Name : "matrix_operations.cuh"
// Title : matrix_operations
// Date : 11/09/2026
// Version : 1.0
// Target Hardware : NVIDIA Jetson Orin Nano
// Author : Adam Roccanova
//
// Revision History : Initial version
//
//
//**************************************************************************
#ifndef MATRIX_OPERATIONS_CUH
#define MATRIX_OPERATIONS_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void mulMatrixOnGPUSharedUnroll(float *MatA, float *MatB, float *MatC, int M, int N, int K);   //tiled multiplication using shared memory, with the tile loop fully unrolled
__global__ void matrix_transpose_2D2D (float *MatA, float *MatB, int rows, int cols);                     //transpose with 2D grid and 2D blocks
__global__ void addBias(float *Z, const float *b, int rows, int cols);                                    //matrix addition for bias

#endif
