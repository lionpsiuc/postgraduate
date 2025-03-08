/**
 * @file matrix.c
 *
 * @brief Implementation of CPU matrix operations and utility functions.
 *
 * This file contains implementations for matrix operations performed on the
 * CPU, including allocation, deallocation, row-wise and column-wise sum
 * calculations, reduction operations, and results output functionality.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-06
 * @version 1.0
 */

#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Allocates and initialises a matrix with random float values.
 *
 * Allocates a 2D matrix of size n x m and initialises each element with a
 * random float value between -20.0 and 20.0 using drand48.
 *
 * @param[in] n Number of rows in the matrix.
 * @param[in] m Number of columns in the matrix.
 *
 * @returns Pointer to the allocated and initialised n x m matrix.
 */
float **allocateMatrix(int n, int m) {
  float **matrix = (float **)malloc(n * sizeof(float *));
  if (!matrix) {
    fprintf(stderr, "Error: Memory allocation failed for matrix rows\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < n; i++) {
    matrix[i] = (float *)malloc(m * sizeof(float));
    if (!matrix[i]) {
      fprintf(stderr, "Error: Memory allocation failed for matrix column %d\n",
              i);

      // Free previously allocated memory
      for (int j = 0; j < i; j++) {
        free(matrix[j]);
      }
      free(matrix);
      exit(EXIT_FAILURE);
    }

    // Initialise with random values
    for (int j = 0; j < m; j++) {
      matrix[i][j] = ((float)(drand48()) * 40.0f) - 20.0f;
    }
  }

  // Return the matrix
  return matrix;
}

/**
 * @brief Deallocates memory used by a matrix.
 *
 * Frees all memory associated with the matrix, including each row and the array
 * of row pointers.
 *
 * @param[in] matrix Pointer to the matrix to be freed.
 * @param[in] n Number of rows in the matrix.
 */
void freeMatrix(float **matrix, int n) {
  if (matrix) {
    for (int i = 0; i < n; i++) {
      if (matrix[i]) {
        free(matrix[i]);
      }
    }
    free(matrix);
  }
}

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
float *computeRowSums(float **matrix, int n, int m) {
  float *rowSums = (float *)malloc(n * sizeof(float));
  if (!rowSums) {
    fprintf(stderr, "Error: Memory allocation failed for row sums\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < n; i++) {
    rowSums[i] = 0.0f;
    for (int j = 0; j < m; j++) {
      rowSums[i] += fabsf(matrix[i][j]);
    }
  }
  return rowSums;
}

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
float *computeColumnSums(float **matrix, int n, int m) {
  float *colSums = (float *)malloc(m * sizeof(float));
  if (!colSums) {
    fprintf(stderr, "Error: Memory allocation failed for column sums\n");
    exit(EXIT_FAILURE);
  }
  for (int j = 0; j < m; j++) {
    colSums[j] = 0.0f;
    for (int i = 0; i < n; i++) {
      colSums[j] += fabsf(matrix[i][j]);
    }
  }
  return colSums;
}

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
float reduce(float *vector, int size) {
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += vector[i];
  }
  return sum;
}

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
void writeResults(const char *filename, int n, int m, int threads_per_block,
                  double cpu_row_time, double cpu_col_time,
                  double cpu_reduce_row_time, double cpu_reduce_col_time,
                  double gpu_row_time, double gpu_col_time,
                  double gpu_reduce_row_time, double gpu_reduce_col_time,
                  float cpu_row_sum, float cpu_col_sum, float gpu_row_sum,
                  float gpu_col_sum) {
  FILE *file = fopen(filename, "a");
  if (!file) {
    fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
    return;
  }

  // Check if the file is empty, and if so, add a header
  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  if (size == 0) {
    fprintf(file,
            "n,m,threads_per_block,"
            "cpu_row_time,cpu_col_time,cpu_reduce_row_time,cpu_reduce_col_time,"
            "gpu_row_time,gpu_col_time,gpu_reduce_row_time,gpu_reduce_col_time,"
            "cpu_row_sum,cpu_col_sum,gpu_row_sum,gpu_col_sum,"
            "row_speedup,col_speedup,row_reduce_speedup,col_reduce_speedup,"
            "row_error,col_error\n");
  }

  // Calculate speedups
  double row_speedup = cpu_row_time / gpu_row_time;
  double col_speedup = cpu_col_time / gpu_col_time;
  double row_reduce_speedup = cpu_reduce_row_time / gpu_reduce_row_time;
  double col_reduce_speedup = cpu_reduce_col_time / gpu_reduce_col_time;

  // Calculate relative errors
  float row_error = fabsf(cpu_row_sum - gpu_row_sum) / fabsf(cpu_row_sum);
  float col_error = fabsf(cpu_col_sum - gpu_col_sum) / fabsf(cpu_col_sum);

  // Write results
  fprintf(file,
          "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%."
          "6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.8f,%.8f\n",
          n, m, threads_per_block, cpu_row_time, cpu_col_time,
          cpu_reduce_row_time, cpu_reduce_col_time, gpu_row_time, gpu_col_time,
          gpu_reduce_row_time, gpu_reduce_col_time, cpu_row_sum, cpu_col_sum,
          gpu_row_sum, gpu_col_sum, row_speedup, col_speedup,
          row_reduce_speedup, col_reduce_speedup, row_error, col_error);

  fclose(file);
}
