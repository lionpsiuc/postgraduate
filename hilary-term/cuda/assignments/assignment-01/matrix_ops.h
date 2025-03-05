#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

// Function to allocate and initialize a matrix
float **allocateMatrix(int n, int m);

// Function to free a matrix
void freeMatrix(float **matrix, int n);

// Function to add absolute values of each row
float *computeRowSums(float **matrix, int n, int m);

// Function to add absolute values of each column
float *computeColumnSums(float **matrix, int n, int m);

// Function to reduce a vector to a single value by summing
float reduce(float *vector, int size);

// Function to print a matrix (for debugging)
void printMatrix(float **matrix, int n, int m);

// Function to print a vector (for debugging)
void printVector(float *vector, int size);

// Function to write performance results to a file
void writeResults(const char *filename, int n, int m, int threads_per_block,
                  double cpu_row_time, double cpu_col_time,
                  double cpu_reduce_row_time, double cpu_reduce_col_time,
                  double gpu_row_time, double gpu_col_time,
                  double gpu_reduce_row_time, double gpu_reduce_col_time,
                  float cpu_row_sum, float cpu_col_sum, float gpu_row_sum,
                  float gpu_col_sum);

#endif // MATRIX_OPS_H
