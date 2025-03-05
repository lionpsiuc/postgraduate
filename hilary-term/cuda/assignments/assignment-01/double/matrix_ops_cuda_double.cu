#include "matrix_ops_cuda_double.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  return device;
}

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

// Kernel for computing row sums
__global__ void computeRowSumsKernelDouble(double *matrix_flat, double *rowSums,
                                           int n, int m) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    double sum = 0.0;
    for (int j = 0; j < m; j++) {
      sum += fabs(matrix_flat[row * m + j]);
    }
    rowSums[row] = sum;
  }
}

// Kernel for computing column sums
__global__ void computeColumnSumsKernelDouble(double *matrix_flat,
                                              double *colSums, int n, int m) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (col < m) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      sum += fabs(matrix_flat[i * m + col]);
    }
    colSums[col] = sum;
  }
}

// Kernel for reducing a vector without atomic operations
__global__ void reduceKernelDouble(double *input, double *output, int size) {
  extern __shared__ double sdata[];

  // Each thread loads one element from global memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load into shared memory
  sdata[tid] = (i < size) ? input[i] : 0;
  __syncthreads();

  // Do reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global memory
  if (tid == 0)
    output[blockIdx.x] = sdata[0];
}

// Function to flatten a 2D array for CUDA
double *flattenMatrixDouble(double **matrix, int n, int m) {
  double *flat = (double *)malloc(n * m * sizeof(double));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      flat[i * m + j] = matrix[i][j];
    }
  }
  return flat;
}

// Compute row sums on GPU
extern "C" double *computeRowSumsGPUDouble(double **matrix, int n, int m,
                                           int threads_per_block) {
  // Flatten the matrix for GPU
  double *matrix_flat = flattenMatrixDouble(matrix, n, m);

  // Allocate device memory
  double *d_matrix, *d_rowSums;
  cudaMalloc((void **)&d_matrix, n * m * sizeof(double));
  cudaCheckError();
  cudaMalloc((void **)&d_rowSums, n * sizeof(double));
  cudaCheckError();

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix_flat, n * m * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  // Setup grid and blocks
  int numBlocks = (n + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  computeRowSumsKernelDouble<<<numBlocks, threads_per_block>>>(d_matrix,
                                                               d_rowSums, n, m);
  cudaCheckError();

  // Allocate host memory for result
  double *rowSums = (double *)malloc(n * sizeof(double));

  // Copy result back to host
  cudaMemcpy(rowSums, d_rowSums, n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_rowSums);
  free(matrix_flat);

  return rowSums;
}

// Compute column sums on GPU
extern "C" double *computeColumnSumsGPUDouble(double **matrix, int n, int m,
                                              int threads_per_block) {
  // Flatten the matrix for GPU
  double *matrix_flat = flattenMatrixDouble(matrix, n, m);

  // Allocate device memory
  double *d_matrix, *d_colSums;
  cudaMalloc((void **)&d_matrix, n * m * sizeof(double));
  cudaCheckError();
  cudaMalloc((void **)&d_colSums, m * sizeof(double));
  cudaCheckError();

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix_flat, n * m * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  // Setup grid and blocks
  int numBlocks = (m + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  computeColumnSumsKernelDouble<<<numBlocks, threads_per_block>>>(
      d_matrix, d_colSums, n, m);
  cudaCheckError();

  // Allocate host memory for result
  double *colSums = (double *)malloc(m * sizeof(double));

  // Copy result back to host
  cudaMemcpy(colSums, d_colSums, m * sizeof(double), cudaMemcpyDeviceToHost);
  cudaCheckError();

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_colSums);
  free(matrix_flat);

  return colSums;
}

// Reduce vector on GPU without atomic operations
extern "C" double reduceGPUDouble(double *vector, int size,
                                  int threads_per_block) {
  // Allocate device memory
  double *d_vector, *d_partial_sums;

  // Number of blocks needed for the first reduction pass
  int numBlocks = (size + threads_per_block - 1) / threads_per_block;

  cudaMalloc((void **)&d_vector, size * sizeof(double));
  cudaCheckError();
  cudaMalloc((void **)&d_partial_sums, numBlocks * sizeof(double));
  cudaCheckError();

  // Copy vector to device
  cudaMemcpy(d_vector, vector, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaCheckError();

  // First reduction pass - reduce input array to partial sums
  reduceKernelDouble<<<numBlocks, threads_per_block,
                       threads_per_block * sizeof(double)>>>(
      d_vector, d_partial_sums, size);
  cudaCheckError();

  // If we have more than one block, we need to reduce the partial sums
  double result = 0.0;

  if (numBlocks > 1) {
    // Allocate CPU memory for partial sums
    double *partial_sums = (double *)malloc(numBlocks * sizeof(double));

    // Copy partial sums back to host
    cudaMemcpy(partial_sums, d_partial_sums, numBlocks * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Final reduction on CPU (for simplicity)
    for (int i = 0; i < numBlocks; i++) {
      result += partial_sums[i];
    }

    free(partial_sums);
  } else {
    // If only one block was used, just copy the single result
    cudaMemcpy(&result, d_partial_sums, sizeof(double), cudaMemcpyDeviceToHost);
    cudaCheckError();
  }

  // Free device memory
  cudaFree(d_vector);
  cudaFree(d_partial_sums);

  return result;
}
