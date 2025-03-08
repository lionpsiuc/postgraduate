/**
 * @file matrix-cuda-double.h
 *
 * @brief Header file for GPU matrix operations using CUDA.
 *
 * This file contains function declarations for parallel matrix operations
 * implemented with CUDA, including row-wise and column-wise sum calculations,
 * and reduction operations.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-06
 * @version 1.0
 */

#ifndef MATRIX_CUDA_DOUBLE_H
#define MATRIX_CUDA_DOUBLE_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Computes the sum of absolute values for each row in the matrix using
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
double *computeRowSumsGPUDouble(double **matrix, int n, int m,
                                int threads_per_block);

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
double *computeColumnSumsGPUDouble(double **matrix, int n, int m,
                                   int threads_per_block);

/**
 * @brief Reduces a vector to a single value by summing all its elements using
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
double reduceGPUDouble(double *vector, int size, int threads_per_block);

/**
 * @brief Initialises the CUDA environment and selects a device.
 *
 * Checks for available CUDA devices, selects the first available device,
 * and prints device information. Returns error codes if no devices are
 * available.
 *
 * @returns Device index on success, or -1 on failure.
 */
int setupCuda();

#ifdef __cplusplus
}
#endif

#endif // MATRIX_CUDA_DOUBLE_H
