#include "matrix_ops_double.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Function to allocate and initialize a matrix
double** allocateMatrixDouble(int n, int m) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    if (!matrix) {
        fprintf(stderr, "Error: Memory allocation failed for matrix rows\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(m * sizeof(double));
        if (!matrix[i]) {
            fprintf(stderr, "Error: Memory allocation failed for matrix column %d\n", i);
            // Free previously allocated memory
            for (int j = 0; j < i; j++) {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }

        // Initialize with random values between -20 and 20
        for (int j = 0; j < m; j++) {
            matrix[i][j] = ((double)(drand48()) * 40.0) - 20.0;
        }
    }

    return matrix;
}

// Function to free a matrix
void freeMatrixDouble(double** matrix, int n) {
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
double* computeRowSumsDouble(double** matrix, int n, int m) {
    double* rowSums = (double*)malloc(n * sizeof(double));
    if (!rowSums) {
        fprintf(stderr, "Error: Memory allocation failed for row sums\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        rowSums[i] = 0.0;
        for (int j = 0; j < m; j++) {
            rowSums[i] += fabs(matrix[i][j]);
        }
    }

    return rowSums;
}

// Function to add absolute values of each column
double* computeColumnSumsDouble(double** matrix, int n, int m) {
    double* colSums = (double*)malloc(m * sizeof(double));
    if (!colSums) {
        fprintf(stderr, "Error: Memory allocation failed for column sums\n");
        exit(EXIT_FAILURE);
    }

    for (int j = 0; j < m; j++) {
        colSums[j] = 0.0;
        for (int i = 0; i < n; i++) {
            colSums[j] += fabs(matrix[i][j]);
        }
    }

    return colSums;
}

// Function to reduce a vector to a single value by summing
double reduceDouble(double* vector, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += vector[i];
    }
    return sum;
}

// Function to print a matrix (for debugging)
void printMatrixDouble(double** matrix, int n, int m) {
    printf("Matrix (%d x %d):\n", n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%8.2f ", matrix[i][j]);
            if (j % 10 == 9) printf("\n");
        }
        printf("\n");
    }
}

// Function to print a vector (for debugging)
void printVectorDouble(double* vector, int size) {
    printf("Vector (size %d):\n", size);
    for (int i = 0; i < size; i++) {
        printf("%8.2f ", vector[i]);
        if (i % 10 == 9) printf("\n");
    }
    printf("\n");
}

// Function to write performance results to a file
void writeResultsDouble(const char* filename, int n, int m, int threads_per_block, 
                  double cpu_row_time, double cpu_col_time, double cpu_reduce_row_time, double cpu_reduce_col_time,
                  double gpu_row_time, double gpu_col_time, double gpu_reduce_row_time, double gpu_reduce_col_time,
                  double cpu_row_sum, double cpu_col_sum, double gpu_row_sum, double gpu_col_sum) {
                      
    FILE* file = fopen(filename, "a");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file %s for writing\n", filename);
        return;
    }
    
    // Check if the file is empty, if so add a header
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    if (size == 0) {
        fprintf(file, "precision,n,m,threads_per_block,"
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
    double row_error = fabs(cpu_row_sum - gpu_row_sum) / fabs(cpu_row_sum);
    double col_error = fabs(cpu_col_sum - gpu_col_sum) / fabs(cpu_col_sum);
    
    // Write results
    fprintf(file, "double,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.2f,%.2f,%.2f,%.8f,%.8f\n",
            n, m, threads_per_block,
            cpu_row_time, cpu_col_time, cpu_reduce_row_time, cpu_reduce_col_time,
            gpu_row_time, gpu_col_time, gpu_reduce_row_time, gpu_reduce_col_time,
            cpu_row_sum, cpu_col_sum, gpu_row_sum, gpu_col_sum,
            row_speedup, col_speedup, row_reduce_speedup, col_reduce_speedup,
            row_error, col_error);
    
    fclose(file);
}