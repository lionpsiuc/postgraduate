#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <getopt.h>
#include "matrix_ops.h"
#include "matrix_ops_cuda.h"

// Function to print usage information
void printUsage(char* programName) {
    printf("Usage: %s [options]\n", programName);
    printf("Options:\n");
    printf("  -n <size>      Row size of the matrix (default: 10)\n");
    printf("  -m <size>      Column size of the matrix (default: 10)\n");
    printf("  -b <size>      Number of threads per block for CUDA (default: 256)\n");
    printf("  -r             Use current time as random seed (default: fixed seed 1234567)\n");
    printf("  -t             Display timing information\n");
    printf("  -o <filename>  Write results to file (default: results.csv)\n");
    printf("  -h             Display this help message\n");
    printf("  -v             Verbose output (print matrix and vectors)\n");
}

int main(int argc, char* argv[]) {
    // Default parameters
    int n = 10;                  // Default row size
    int m = 10;                  // Default column size
    int threads_per_block = 256; // Default threads per block
    int randomSeed = 0;          // Use fixed seed by default
    int showTiming = 0;          // Don't show timing by default
    int verbose = 0;             // Don't print matrix by default
    char outputFilename[256] = "results.csv"; // Default output file
    
    // Parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "n:m:b:rto:hv")) != -1) {
        switch (opt) {
            case 'n':
                n = atoi(optarg);
                break;
            case 'm':
                m = atoi(optarg);
                break;
            case 'b':
                threads_per_block = atoi(optarg);
                break;
            case 'r':
                randomSeed = 1;
                break;
            case 't':
                showTiming = 1;
                break;
            case 'o':
                strncpy(outputFilename, optarg, sizeof(outputFilename)-1);
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            case 'v':
                verbose = 1;
                break;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }
    
    // Setup random seed
    if (randomSeed) {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        srand48((int)(tv.tv_usec));
    } else {
        srand48(1234567);
    }
    
    printf("Matrix size: %d x %d\n", n, m);
    printf("Threads per block: %d\n", threads_per_block);
    
    // Initialize CUDA
    setupCuda();
    
    // Allocate and initialize matrix
    printf("Allocating and initializing matrix...\n");
    float** matrix = allocateMatrix(n, m);
    
    if (verbose) {
        printMatrix(matrix, n, m);
    }
    
    // Variables for timing
    struct timespec start, end;
    double cpu_row_time, cpu_col_time, cpu_reduce_row_time, cpu_reduce_col_time;
    double gpu_row_time, gpu_col_time, gpu_reduce_row_time, gpu_reduce_col_time;
    
    // CPU computations
    printf("Performing CPU computations...\n");
    
    // Row sums
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* rowSums = computeRowSums(matrix, n, m);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cpu_row_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    if (verbose) {
        printVector(rowSums, n);
    }
    
    // Row reduction
    clock_gettime(CLOCK_MONOTONIC, &start);
    float rowSum = reduce(rowSums, n);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cpu_reduce_row_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("CPU row sum: %f\n", rowSum);
    
    // Column sums
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* colSums = computeColumnSums(matrix, n, m);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cpu_col_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    if (verbose) {
        printVector(colSums, m);
    }
    
    // Column reduction
    clock_gettime(CLOCK_MONOTONIC, &start);
    float colSum = reduce(colSums, m);
    clock_gettime(CLOCK_MONOTONIC, &end);
    cpu_reduce_col_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("CPU column sum: %f\n", colSum);
    
    // GPU computations
    printf("Performing GPU computations...\n");
    
    // Row sums
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* rowSumsGPU = computeRowSumsGPU(matrix, n, m, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_row_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Row reduction
    clock_gettime(CLOCK_MONOTONIC, &start);
    float rowSumGPU = reduceGPU(rowSumsGPU, n, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_reduce_row_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("GPU row sum: %f\n", rowSumGPU);
    
    // Column sums
    clock_gettime(CLOCK_MONOTONIC, &start);
    float* colSumsGPU = computeColumnSumsGPU(matrix, n, m, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_col_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    // Column reduction
    clock_gettime(CLOCK_MONOTONIC, &start);
    float colSumGPU = reduceGPU(colSumsGPU, m, threads_per_block);
    clock_gettime(CLOCK_MONOTONIC, &end);
    gpu_reduce_col_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("GPU column sum: %f\n", colSumGPU);
    
    // Display timing information if requested
    if (showTiming) {
        printf("\nTiming Information:\n");
        printf("------------------\n");
        printf("CPU row sum computation time: %f seconds\n", cpu_row_time);
        printf("CPU column sum computation time: %f seconds\n", cpu_col_time);
        printf("CPU row reduction time: %f seconds\n", cpu_reduce_row_time);
        printf("CPU column reduction time: %f seconds\n", cpu_reduce_col_time);
        
        printf("GPU row sum computation time: %f seconds\n", gpu_row_time);
        printf("GPU column sum computation time: %f seconds\n", gpu_col_time);
        printf("GPU row reduction time: %f seconds\n", gpu_reduce_row_time);
        printf("GPU column reduction time: %f seconds\n", gpu_reduce_col_time);
        
        printf("\nSpeedups:\n");
        printf("Row sum speedup: %f\n", cpu_row_time / gpu_row_time);
        printf("Column sum speedup: %f\n", cpu_col_time / gpu_col_time);
        printf("Row reduction speedup: %f\n", cpu_reduce_row_time / gpu_reduce_row_time);
        printf("Column reduction speedup: %f\n", cpu_reduce_col_time / gpu_reduce_col_time);
        
        printf("\nRelative Errors:\n");
        printf("Row sum relative error: %e\n", fabsf(rowSum - rowSumGPU) / fabsf(rowSum));
        printf("Column sum relative error: %e\n", fabsf(colSum - colSumGPU) / fabsf(colSum));
    }
    
    // Write results to file
    writeResults(outputFilename, n, m, threads_per_block,
                 cpu_row_time, cpu_col_time, cpu_reduce_row_time, cpu_reduce_col_time,
                 gpu_row_time, gpu_col_time, gpu_reduce_row_time, gpu_reduce_col_time,
                 rowSum, colSum, rowSumGPU, colSumGPU);
    
    // Free memory
    free(rowSums);
    free(colSums);
    free(rowSumsGPU);
    free(colSumsGPU);
    freeMatrix(matrix, n);
    
    return 0;
}