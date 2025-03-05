#ifndef MATRIX_OPS_CUDA_H
#define MATRIX_OPS_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Function to perform row sums on GPU
float *computeRowSumsGPU(float **matrix, int n, int m, int threads_per_block);

// Function to perform column sums on GPU
float *computeColumnSumsGPU(float **matrix, int n, int m,
                            int threads_per_block);

// Function to perform reduction on GPU
float reduceGPU(float *vector, int size, int threads_per_block);

// Function to setup CUDA device
int setupCuda();

#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPS_CUDA_H
