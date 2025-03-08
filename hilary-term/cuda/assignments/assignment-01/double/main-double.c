/**
 * @file main-double.c
 *
 * @brief Main programme for matrix operations benchmark in double precision.
 *
 * This programme implements the first, second, and third task of the
 * assignment, computing row and column sums of a matrix and their reductions on
 * both the CPU and GPU.
 *
 * @author Ion Lipsiuc
 * @date 2025-03-06
 * @version 1.0
 */

#include "matrix-cuda-double.h"
#include "matrix-double.h"
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

/**
 * @brief Main function.
 *
 * Parses command-line arguments, allocates matrices, performs CPU and GPU
 * calculations, times operations, compares results, and outputs benchmark data.
 *
 * -n <size>: Number of rows, with the default being 10.
 * -m <size>: Number of columns, with the default being 10.
 * -b <num>: Threads per block for GPU, with the default being 256.
 * -c: Runs the programme only on the CPU, as is required for the first task.
 * -r: Random seed based on current time instead of fixed seed.
 * -t: Display timing information.
 * -o <file>: Write benchmark results to a CSV file.
 *
 * @param[in] argc Number of command-line arguments.
 * @param[in] argv Array of command-line argument strings.
 *
 * @returns 0 upon successful execution.
 */
int main(int argc, char *argv[]) {

  // Default parameters
  int n = 10;                    // Default row size
  int m = 10;                    // Default column size
  int threads_per_block = 256;   // Default threads per block
  int randomSeed = 0;            // Use fixed seed by default
  int showTiming = 0;            // Don't show timing by default
  int cpuOnly = 0;               // Flag to only use CPU
  int writeToFile = 0;           // Don't write to file by default
  char outputFilename[256] = ""; // Empty default output file

  // Parse command-line arguments
  int opt;
  while ((opt = getopt(argc, argv, "b:cm:n:o:rt")) != -1) {
    switch (opt) {
    case 'b':
      threads_per_block = atoi(optarg);
      break;
    case 'c':
      cpuOnly = 1;
      break;
    case 'm':
      m = atoi(optarg);
      break;
    case 'n':
      n = atoi(optarg);
      break;
    case 'o':
      strncpy(outputFilename, optarg, sizeof(outputFilename) - 1);
      writeToFile = 1;
      break;
    case 'r':
      randomSeed = 1;
      break;
    case 't':
      showTiming = 1;
      break;
    }
  }

  // Seed
  if (randomSeed) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    srand48((int)(tv.tv_usec));
  } else {
    srand48(1234567);
  }

  printf("Using double precision\n");
  printf("Matrix size: %d x %d\n", n, m);

  // If not only using the CPU, display threads per block
  if (!cpuOnly) {
    printf("Threads per block: %d\n", threads_per_block);

    // Initialise CUDA
    setupCuda();
  } else { // Only using the CPU
    printf("Only using the CPU\n");
  }

  // Allocate and initialise matrix
  printf("Allocating and initialising matrix...\n");
  double **matrix = allocateMatrixDouble(n, m);

  // Variables for timing
  struct timespec start, end;
  double cpu_row_time = 0.0, cpu_col_time = 0.0;
  double cpu_reduce_row_time = 0.0, cpu_reduce_col_time = 0.0;
  double gpu_row_time = 0.0, gpu_col_time = 0.0;
  double gpu_reduce_row_time = 0.0, gpu_reduce_col_time = 0.0;

  // Initialise CPU result variables
  double rowSum = 0.0;
  double colSum = 0.0;
  double *rowSums = NULL;
  double *colSums = NULL;

  // CPU computations
  printf("Performing CPU computations...\n");

  // Row sums
  clock_gettime(CLOCK_MONOTONIC, &start);
  rowSums = computeRowSumsDouble(matrix, n, m);
  clock_gettime(CLOCK_MONOTONIC, &end);
  cpu_row_time =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // Row reduction
  clock_gettime(CLOCK_MONOTONIC, &start);
  rowSum = reduceDouble(rowSums, n);
  clock_gettime(CLOCK_MONOTONIC, &end);
  cpu_reduce_row_time =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // Print result
  printf("CPU row sum: %f\n", rowSum);

  // Column sums
  clock_gettime(CLOCK_MONOTONIC, &start);
  colSums = computeColumnSumsDouble(matrix, n, m);
  clock_gettime(CLOCK_MONOTONIC, &end);
  cpu_col_time =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // Column reduction
  clock_gettime(CLOCK_MONOTONIC, &start);
  colSum = reduceDouble(colSums, m);
  clock_gettime(CLOCK_MONOTONIC, &end);
  cpu_reduce_col_time =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // Print result
  printf("CPU column sum: %f\n", colSum);

  // Initialise GPU result variables
  double rowSumGPU = 0.0;
  double colSumGPU = 0.0;
  double *rowSumsGPU = NULL;
  double *colSumsGPU = NULL;

  // GPU computations
  if (!cpuOnly) {
    printf("Performing GPU computations...\n");

    // Row sums
    clock_gettime(CLOCK_MONOTONIC, &start);
    rowSumsGPU = computeRowSumsGPUDouble(matrix, n, m, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_row_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Row reduction
    clock_gettime(CLOCK_MONOTONIC, &start);
    rowSumGPU = reduceGPUDouble(rowSumsGPU, n, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_reduce_row_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Print result
    printf("GPU row sum: %f\n", rowSumGPU);

    // Column sums
    clock_gettime(CLOCK_MONOTONIC, &start);
    colSumsGPU = computeColumnSumsGPUDouble(matrix, n, m, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_col_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Column reduction
    clock_gettime(CLOCK_MONOTONIC, &start);
    colSumGPU = reduceGPUDouble(colSumsGPU, m, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_reduce_col_time =
        (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Print result
    printf("GPU column sum: %f\n", colSumGPU);

  } else {

    // When only using the CPU, we don't need to set these values
    rowSumGPU = 0.0;
    colSumGPU = 0.0;

    // No need to allocate memory
    rowSumsGPU = NULL;
    colSumsGPU = NULL;

    // Set GPU times to zero
    gpu_row_time = gpu_col_time = gpu_reduce_row_time = gpu_reduce_col_time =
        0.0;
  }

  // Display timing information if requested
  if (showTiming) {
    printf("\nTiming Information:\n");
    printf("CPU row sum computation time: %f seconds\n", cpu_row_time);
    printf("CPU column sum computation time: %f seconds\n", cpu_col_time);
    printf("CPU row reduction time: %f seconds\n", cpu_reduce_row_time);
    printf("CPU column reduction time: %f seconds\n", cpu_reduce_col_time);

    // Display GPU results
    if (!cpuOnly) {

      // Computation times
      printf("GPU row sum computation time: %f seconds\n", gpu_row_time);
      printf("GPU column sum computation time: %f seconds\n", gpu_col_time);
      printf("GPU row reduction time: %f seconds\n", gpu_reduce_row_time);
      printf("GPU column reduction time: %f seconds\n", gpu_reduce_col_time);

      // Speedups
      printf("\nSpeedups:\n");
      printf("Row sum speedup: %f\n", cpu_row_time / gpu_row_time);
      printf("Column sum speedup: %f\n", cpu_col_time / gpu_col_time);
      printf("Row reduction speedup: %f\n",
             cpu_reduce_row_time / gpu_reduce_row_time);
      printf("Column reduction speedup: %f\n",
             cpu_reduce_col_time / gpu_reduce_col_time);

      // Relative errors
      printf("\nRelative Errors:\n");
      printf("Row sum relative error: %e\n",
             fabs(rowSum - rowSumGPU) / fabs(rowSum));
      printf("Column sum relative error: %e\n",
             fabs(colSum - colSumGPU) / fabs(colSum));
    }
  }

  // Write results to file so long as we're not only using the CPU and the
  // relevant flag is enabled
  if (!cpuOnly && writeToFile) {
    writeResultsDouble(outputFilename, n, m, threads_per_block, cpu_row_time,
                       cpu_col_time, cpu_reduce_row_time, cpu_reduce_col_time,
                       gpu_row_time, gpu_col_time, gpu_reduce_row_time,
                       gpu_reduce_col_time, rowSum, colSum, rowSumGPU,
                       colSumGPU);
  }

  // Free memory
  free(rowSums);
  free(colSums);
  if (rowSumsGPU)
    free(rowSumsGPU);
  if (colSumsGPU)
    free(colSumsGPU);
  freeMatrixDouble(matrix, n);

  return 0;
}
