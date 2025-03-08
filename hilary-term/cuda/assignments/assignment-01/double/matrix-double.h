/**
 * @file matrix-double.h
 *
 * @brief Header file for CPU matrix operations and utility functions.
 *
 * This file contains function declarations for matrix operations performed on
 * the CPU, including allocation, deallocation, row-wise and column-wise sum
 * calculations, reduction operations, and results output functionality.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-06
 * @version 1.0
 */

#ifndef MATRIX_DOUBLE_H
#define MATRIX_DOUBLE_H

/**
 * @brief Allocates and initialises a matrix with random double values.
 *
 * Allocates a 2D matrix of size n x m and initialises each element with a
 * random double value between -20.0 and 20.0 using drand48.
 *
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 *
 * @returns Pointer to the allocated and initialised n x m matrix.
 */
double **allocateMatrixDouble(int n, int m);

/**
 * @brief Deallocates memory used by a matrix.
 *
 * Frees all memory associated with the matrix, including each row and the array
 * of row pointers.
 *
 * @param[in] matrix Pointer to the matrix to be freed.
 * @param[in] n Number of rows in the matrix.
 */
void freeMatrixDouble(double **matrix, int n);

/**
 * @brief Computes the sum of absolute values for each row in the matrix.
 *
 * For each row, calculates the sum of absolute values of all elements in that
 * row.
 *
 * @param[in] matrix Input matrix to process.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 *
 * @returns Array containing the sum of absolute values for each row.
 */
double *computeRowSumsDouble(double **matrix, int n, int m);

/**
 * @brief Computes the sum of absolute values for each column in the matrix.
 *
 * For each column, calculates the sum of absolute values of all elements in
 * that column.
 *
 * @param[in] matrix Input matrix to process.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 *
 * @returns Array containing the sum of absolute values for each column.
 */
double *computeColumnSumsDouble(double **matrix, int n, int m);

/**
 * @brief Reduces a vector to a single value by summing all its elements.
 *
 * Takes a vector and computes the sum of all its elements.
 *
 * @param[in] vector Input vector to be reduced.
 * @param[in] size Number of elements in the vector.
 *
 * @returns Sum of all elements in the vector.
 */
double reduceDouble(double *vector, int size);

/**
 * @brief Writes benchmark results to a CSV file.
 *
 * Records performance metrics including timing, sums, speedups, and error
 * measurements for both CPU and GPU implementations of matrix operations.
 *
 * @param[in] filename Name of the output CSV file.
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 * @param[in] threads_per_block Number of threads per block used in GPU
 * computation.
 * @param[in] cpu_row_time Time taken by CPU to compute row sums.
 * @param[in] cpu_col_time Time taken by CPU to compute column sums.
 * @param[in] cpu_reduce_row_time Time taken by CPU to reduce row sums.
 * @param[in] cpu_reduce_col_time Time taken by CPU to reduce column sums.
 * @param[in] gpu_row_time Time taken by GPU to compute row sums.
 * @param[in] gpu_col_time Time taken by GPU to compute column sums.
 * @param[in] gpu_reduce_row_time Time taken by GPU to reduce row sums.
 * @param[in] gpu_reduce_col_time Time taken by GPU to reduce column sums.
 * @param[in] cpu_row_sum Result of CPU row sums reduction.
 * @param[in] cpu_col_sum Result of CPU column sums reduction.
 * @param[in] gpu_row_sum Result of GPU row sums reduction.
 * @param[in] gpu_col_sum Result of GPU column sums reduction.
 */
void writeResultsDouble(const char *filename, int n, int m,
                        int threads_per_block, double cpu_row_time,
                        double cpu_col_time, double cpu_reduce_row_time,
                        double cpu_reduce_col_time, double gpu_row_time,
                        double gpu_col_time, double gpu_reduce_row_time,
                        double gpu_reduce_col_time, double cpu_row_sum,
                        double cpu_col_sum, double gpu_row_sum,
                        double gpu_col_sum);

#endif // MATRIX_DOUBLE_H
