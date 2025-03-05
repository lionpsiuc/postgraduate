#ifndef MATRIX_OPS_DOUBLE_H
#define MATRIX_OPS_DOUBLE_H

// Function to allocate and initialize a matrix
double **allocateMatrixDouble(int n, int m);

// Function to free a matrix
void freeMatrixDouble(double **matrix, int n);

// Function to add absolute values of each row
double *computeRowSumsDouble(double **matrix, int n, int m);

// Function to add absolute values of each column
double *computeColumnSumsDouble(double **matrix, int n, int m);

// Function to reduce a vector to a single value by summing
double reduceDouble(double *vector, int size);

// Function to print a matrix (for debugging)
void printMatrixDouble(double **matrix, int n, int m);

// Function to print a vector (for debugging)
void printVectorDouble(double *vector, int size);

// Function to write performance results to a file
void writeResultsDouble(const char *filename, int n, int m,
                        int threads_per_block, double cpu_row_time,
                        double cpu_col_time, double cpu_reduce_row_time,
                        double cpu_reduce_col_time, double gpu_row_time,
                        double gpu_col_time, double gpu_reduce_row_time,
                        double gpu_reduce_col_time, double cpu_row_sum,
                        double cpu_col_sum, double gpu_row_sum,
                        double gpu_col_sum);

#endif // MATRIX_OPS_DOUBLE_H
