#include "matrix_ops_cuda.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define cudaCheckError()                                                       \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

// Setup the CUDA device
extern "C" int setupCuda() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    fprintf(stderr, "Error: No CUDA devices found\n");
    return -1;
  }

  int device = 0; // Use the first device by default
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);
  printf("Using CUDA device: %s\n", deviceProp.name);

  cudaSetDevice(device);
  cudaCheckError();

  return device;
}

// Kernel for computing row sums
__global__ void computeRowSumsKernel(float *matrix_flat, float *rowSums, int n,
                                     int m) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    float sum = 0.0f;
    for (int j = 0; j < m; j++) {
      sum += fabsf(matrix_flat[row * m + j]);
    }
    rowSums[row] = sum;
  }
}

// Kernel for computing column sums
__global__ void computeColumnSumsKernel(float *matrix_flat, float *colSums,
                                        int n, int m) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < m) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += fabsf(matrix_flat[i * m + col]);
    }
    colSums[col] = sum;
  }
}

// Kernel for reducing a vector
__global__ void reduceKernel(float *vector, float *result, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    atomicAdd(result, vector[tid]);
  }
}

// Function to flatten a 2D array for CUDA
float *flattenMatrix(float **matrix, int n, int m) {
  float *flat = (float *)malloc(n * m * sizeof(float));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      flat[i * m + j] = matrix[i][j];
    }
  }
  return flat;
}

// Compute row sums on GPU
extern "C" float *computeRowSumsGPU(float **matrix, int n, int m,
                                    int threads_per_block) {
  // Flatten the matrix for GPU
  float *matrix_flat = flattenMatrix(matrix, n, m);

  // Allocate device memory
  float *d_matrix, *d_rowSums;
  cudaMalloc((void **)&d_matrix, n * m * sizeof(float));
  cudaCheckError();
  cudaMalloc((void **)&d_rowSums, n * sizeof(float));
  cudaCheckError();

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix_flat, n * m * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  // Setup grid and blocks
  int numBlocks = (n + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  computeRowSumsKernel<<<numBlocks, threads_per_block>>>(d_matrix, d_rowSums, n,
                                                         m);
  cudaCheckError();

  // Allocate host memory for result
  float *rowSums = (float *)malloc(n * sizeof(float));

  // Copy result back to host
  cudaMemcpy(rowSums, d_rowSums, n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_rowSums);
  free(matrix_flat);

  return rowSums;
}

// Compute column sums on GPU
extern "C" float *computeColumnSumsGPU(float **matrix, int n, int m,
                                       int threads_per_block) {
  // Flatten the matrix for GPU
  float *matrix_flat = flattenMatrix(matrix, n, m);

  // Allocate device memory
  float *d_matrix, *d_colSums;
  cudaMalloc((void **)&d_matrix, n * m * sizeof(float));
  cudaCheckError();
  cudaMalloc((void **)&d_colSums, m * sizeof(float));
  cudaCheckError();

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix_flat, n * m * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  // Setup grid and blocks
  int numBlocks = (m + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  computeColumnSumsKernel<<<numBlocks, threads_per_block>>>(d_matrix, d_colSums,
                                                            n, m);
  cudaCheckError();

  // Allocate host memory for result
  float *colSums = (float *)malloc(m * sizeof(float));

  // Copy result back to host
  cudaMemcpy(colSums, d_colSums, m * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_colSums);
  free(matrix_flat);

  return colSums;
}

// Reduce vector on GPU
extern "C" float reduceGPU(float *vector, int size, int threads_per_block) {
  // Allocate device memory
  float *d_vector, *d_result;
  cudaMalloc((void **)&d_vector, size * sizeof(float));
  cudaCheckError();
  cudaMalloc((void **)&d_result, sizeof(float));
  cudaCheckError();

  // Initialize result to 0
  float zero = 0.0f;
  cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();

  // Copy vector to device
  cudaMemcpy(d_vector, vector, size * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError();

  // Setup grid and blocks
  int numBlocks = (size + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  reduceKernel<<<numBlocks, threads_per_block>>>(d_vector, d_result, size);
  cudaCheckError();

  // Copy result back to host
  float result;
  cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Free device memory
  cudaFree(d_vector);
  cudaFree(d_result);

  return result;
}
