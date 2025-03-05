#ifndef MATRIX_OPS_CUDA_DOUBLE_H
#define MATRIX_OPS_CUDA_DOUBLE_H

#ifdef __cplusplus
extern "C" {
#endif

// Function to perform row sums on GPU
double* computeRowSumsGPUDouble(double** matrix, int n, int m, int threads_per_block);

// Function to perform column sums on GPU
double* computeColumnSumsGPUDouble(double** matrix, int n, int m, int threads_per_block);

// Function to perform reduction on GPU
double reduceGPUDouble(double* vector, int size, int threads_per_block);

// Function to setup CUDA device (shared with single precision)
int setupCuda();

#ifdef __cplusplus
}
#endif

#endif // MATRIX_OPS_CUDA_DOUBLE_H