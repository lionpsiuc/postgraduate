#include "matrix_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Function to allocate and initialize a matrix
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

    // Initialize with random values between -20 and 20
    for (int j = 0; j < m; j++) {
      matrix[i][j] = ((float)(drand48()) * 40.0f) - 20.0f;
    }
  }

  return matrix;
}

// Function to free a matrix
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

// Function to add absolute values of each row
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

// Function to add absolute values of each column
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

// Function to reduce a vector to a single value by summing
float reduce(float *vector, int size) {
  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    sum += vector[i];
  }
  return sum;
}

// Function to print a matrix (for debugging)
void printMatrix(float **matrix, int n, int m) {
  printf("Matrix (%d x %d):\n", n, m);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%8.2f ", matrix[i][j]);
      if (j % 10 == 9)
        printf("\n");
    }
    printf("\n");
  }
}

// Function to print a vector (for debugging)
void printVector(float *vector, int size) {
  printf("Vector (size %d):\n", size);
  for (int i = 0; i < size; i++) {
    printf("%8.2f ", vector[i]);
    if (i % 10 == 9)
      printf("\n");
  }
  printf("\n");
}

// Function to write performance results to a file
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

  // Check if the file is empty, if so add a header
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
          "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%."
          "6f,%.2f,%.2f,%.2f,%.2f,%.8f,%.8f\n",
          n, m, threads_per_block, cpu_row_time, cpu_col_time,
          cpu_reduce_row_time, cpu_reduce_col_time, gpu_row_time, gpu_col_time,
          gpu_reduce_row_time, gpu_reduce_col_time, cpu_row_sum, cpu_col_sum,
          gpu_row_sum, gpu_col_sum, row_speedup, col_speedup,
          row_reduce_speedup, col_reduce_speedup, row_error, col_error);

  fclose(file);
}
