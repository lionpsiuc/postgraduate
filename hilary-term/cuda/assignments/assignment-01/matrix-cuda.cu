/**
 * @file matrix-cuda.cu
 *
 * @brief Implementation of GPU matrix operations using CUDA.
 *
 * This file contains CUDA kernel implementations for parallel matrix
 * operations, including row-wise and column-wise sum calculations and
 reduction
 * operations. It manages device memory allocation, data transfers between
 host
 * and device, and kernel launches.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-06
 * @version 1.0
 */

#include "matrix-cuda.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Initialises the CUDA environment and selects a device.
 *
 * Checks for available CUDA devices, selects the first available device,
 * and prints device information. Returns error codes if no devices are
 * available.
 *
 * @returns Device index on success, or -1 on failure.
 */
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
    fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return device;
}

/**
 * @brief CUDA kernel for computing row sums of absolute values.
 *
 * Each thread computes the sum of absolute values for one row of the matrix.
 * The matrix is provided in flattened form (i.e., a 1D array) for easier
 CUDA
 * processing.
 *
 * @param[in] matrix_flat Input matrix in flattened 1D array form.
 * @param[out] rowSums Output array where each element will store the sum for
 * one row.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 */
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

/**
 * @brief CUDA kernel for computing column sums of absolute values.
 *
 * Each thread computes the sum of absolute values for one column of the
 matrix.
 * The matrix is provided in flattened form (i.e., a 1D array) for easier
 CUDA
 * processing.
 *
 * @param[in] matrix_flat Input matrix in flattened 1D array form.
 * @param[out] colSums Output array where each element will store the sum for
 * one column.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 */
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

/**
 * @brief CUDA kernel for parallel reduction of a vector.
 *
 * Implements a block-level reduction where each block processes a segment of
 * the input vector. Only the first thread in each block computes the sum for
 * its segment, and the results are stored in blockResults.
 *
 * @param[in] vector Input vector to be reduced.
 * @param[out] blockResults Output array to store intermediate block-level
 * results.
 * @param[in] size Number of elements in the vector.
 */
__global__ void reduceKernel(float *vector, float *blockResults, int size) {
  int blockId = blockIdx.x;
  int blockSize = blockDim.x;
  int blockStart = blockId * blockSize;

  // Only the root thread in each block does the work
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < blockSize && (blockStart + i) < size; i++) {
      sum += vector[blockStart + i];
    }
    blockResults[blockId] = sum;
  }
}

/**
 * @brief CUDA kernel for performing the final reduction step to combine
 * block-level results.
 *
 * Uses a single-thread approach where only thread 0 in block 0 sums all the
 * block-level partial results into a single value.
 *
 * @param[in] blockResults Array containing the partial sums from each
 block's
 * reduction.
 * @param[in] finalResult Pointer to a single float where the final sum will
 be
 * stored.
 * @param[in] numBlocks Number of blocks (i.e., partial results) to be
 summed.
 */
__global__ void finalReductionKernel(float *blockResults, float *finalResult,
                                     int numBlocks) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
      sum += blockResults[i];
    }
    *finalResult = sum;
  }
}

/**
 * @brief Converts a 2D matrix to a flattened 1D array.
 *
 * Takes a 2D matrix and creates a contiguous 1D array representation
 * for easier CUDA processing.
 *
 * @param[in] matrix Input 2D matrix.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 *
 * @returns Flattened 1D array representation of the input matrix.
 */
float *flattenMatrix(float **matrix, int n, int m) {
  float *flat = (float *)malloc(n * m * sizeof(float));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      flat[i * m + j] = matrix[i][j];
    }
  }
  return flat;
}

/**
 * @brief Computes the sum of absolute values for each row in the matrix
 using
 * the GPU.
 *
 * Parallelises row sum computation across CUDA threads, with each thread
 * processing one row of the matrix.
 *
 * @param[in] matrix Input matrix to process.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 * @param[in] threads_per_block Number of threads per CUDA block.
 *
 * @returns Array containing the sum of absolute values for each row.
 */
extern "C" float *computeRowSumsGPU(float **matrix, int n, int m,
                                    int threads_per_block) {

  // Flatten the matrix for the GPU
  float *matrix_flat = flattenMatrix(matrix, n, m);

  // Allocate device memory
  float *d_matrix, *d_rowSums;
  cudaMalloc((void **)&d_matrix, n * m * sizeof(float));
  cudaMalloc((void **)&d_rowSums, n * sizeof(float));

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix_flat, n * m * sizeof(float),
             cudaMemcpyHostToDevice);

  // Grid and blocks
  int numBlocks = (n + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  computeRowSumsKernel<<<numBlocks, threads_per_block>>>(d_matrix, d_rowSums, n,
                                                         m);

  // Allocate host memory for result
  float *rowSums = (float *)malloc(n * sizeof(float));

  // Copy result back to host
  cudaMemcpy(rowSums, d_rowSums, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_rowSums);
  free(matrix_flat);

  return rowSums;
}

/**
 * @brief Computes the sum of absolute values for each column in the matrix
 * using the GPU.
 *
 * Parallelises column sum computation across CUDA threads, with each thread
 * processing one column of the matrix.
 *
 * @param[in] matrix Input matrix to process.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 * @param[in] threads_per_block Number of threads per CUDA block.
 *
 * @returns Array containing the sum of absolute values for each column.
 */
extern "C" float *computeColumnSumsGPU(float **matrix, int n, int m,
                                       int threads_per_block) {

  // Flatten the matrix for the GPU
  float *matrix_flat = flattenMatrix(matrix, n, m);

  // Allocate device memory
  float *d_matrix, *d_colSums;
  cudaMalloc((void **)&d_matrix, n * m * sizeof(float));
  cudaMalloc((void **)&d_colSums, m * sizeof(float));

  // Copy matrix to device
  cudaMemcpy(d_matrix, matrix_flat, n * m * sizeof(float),
             cudaMemcpyHostToDevice);

  // Grid and blocks
  int numBlocks = (m + threads_per_block - 1) / threads_per_block;

  // Launch kernel
  computeColumnSumsKernel<<<numBlocks, threads_per_block>>>(d_matrix, d_colSums,
                                                            n, m);

  // Allocate host memory for result
  float *colSums = (float *)malloc(m * sizeof(float));

  // Copy result back to host
  cudaMemcpy(colSums, d_colSums, m * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_matrix);
  cudaFree(d_colSums);
  free(matrix_flat);

  return colSums;
}

/**
 * @brief Reduces a vector to a single value by summing all its elements
 using
 * the GPU.
 *
 * Implements a parallel reduction algorithm where each block processes a
 * portion of the input vector, and then results are combined on the CPU.
 *
 * @param[in] vector Input vector to be reduced.
 * @param[in] size Number of elements in the vector.
 * @param[in] threads_per_block Number of threads per CUDA block.
 *
 * @returns Sum of all elements in the vector.
 */
extern "C" float reduceGPU(float *vector, int size, int threads_per_block) {

  // Allocate device memory
  float *d_vector, *d_blockResults;
  cudaMalloc((void **)&d_vector, size * sizeof(float));

  // Calculate the number of blocks needed
  int numBlocks = (size + threads_per_block - 1) / threads_per_block;
  cudaMalloc((void **)&d_blockResults, numBlocks * sizeof(float));

  // Copy vector to device
  cudaMemcpy(d_vector, vector, size * sizeof(float), cudaMemcpyHostToDevice);

  // Launch kernel to process blocks
  reduceKernel<<<numBlocks, threads_per_block>>>(d_vector, d_blockResults,
                                                 size);

  // Allocate memory for final result
  float *d_finalResult;
  cudaMalloc((void **)&d_finalResult, sizeof(float));

  // Launch final reduction with one thread
  finalReductionKernel<<<1, 1>>>(d_blockResults, d_finalResult, numBlocks);

  // Copy single result back to host
  float finalResult;
  cudaMemcpy(&finalResult, d_finalResult, sizeof(float),
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_vector);
  cudaFree(d_blockResults);
  cudaFree(d_finalResult);

  return finalResult;
}
