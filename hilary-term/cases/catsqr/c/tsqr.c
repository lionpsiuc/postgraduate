/**
 * @file tsqr.c
 *
 * @brief Communication-avoiding TSQR factorisation using MPI and LAPACK.
 *
 * This implementation performs a communication-avoiding TSQR factorisation of a
 * matrix that is distributed among four MPI processes. Each process performs a
 * local QR factorisation on its assigned block of rows, and the resulting R
 * factors are gathered and further factored to produce the final R. The
 * corresponding Q factors are updated to obtain the final orthogonal matrix.
 *
 * @author Ion Lipsiuc
 * @date 2025-02-21
 * @version 1.0
 */

#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * @brief Prints a matrix.
 *
 * @param[in] name Name to identify the matrix.
 * @param[in] A Pointer to the matrix which is stored in row-major order.
 * @param[in] rows Number of rows in the matrix.
 * @param[in] cols Number of columns in the matrix.
 */
void print_matrix(const char *name, double *A, int rows, int cols) {
  printf("%s:\n", name);
  printf("[");
  for (int i = 0; i < rows; i++) {
    printf("[");
    for (int j = 0; j < cols; j++) {
      printf("% .4f", A[i * cols + j]);
      if (j < cols - 1)
        printf(" ");
    }
    printf("]");
    if (i < rows - 1)
      printf("\n ");
  }
  printf("]\n\n");
}

/**
 * @brief Perform the TSQR factorisation on a distributed matrix.
 *
 * This function divides the global matrix among the available MPI processes,
 * computes a local QR factorisation on each process, gathers the local R
 * factors, and then performs a global QR factorisation on the stacked R matrix.
 * The final Q factor is obtained by updating the local Q factors using the
 * reduction Q factor computed from the stacked R matrix. Moreover, we print out
 * the matrices of a matrix in order to verify that the factorisation was a
 * success.
 *
 * @param[in] m Total number of rows of the global matrix.
 * @param[in] m Number of columns of the matrix.
 */
void tsqr(int m, int n) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rank of this process
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Total number of processes

  // Ensure that the number of rows is evenly divisible by the number of
  // processes
  if (m % size != 0) {
    if (rank == 0)
      fprintf(stderr,
              "Error: m (%d) must be divisible by number of processes (%d)\n",
              m, size);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  int local_rows = m / size; // Number of rows assigned to each process

  // Allocated memory for the local matrix block
  double *A_local = malloc(local_rows * n * sizeof(double));
  if (!A_local) {
    fprintf(stderr, "Memory allocation error on A_local\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  double *A_global = NULL;

  // Allocate and initialise the global matrix on the root process
  if (rank == 0) {
    A_global = malloc(m * n * sizeof(double));
    if (!A_global) {
      fprintf(stderr, "Memory allocation error on A_global\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    srand(time(NULL));

    // Fill the global matrix with random values
    for (int i = 0; i < m * n; i++) {
      A_global[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
  }

  // Scatter the global matrix to all processes
  MPI_Scatter(A_global, local_rows * n, MPI_DOUBLE, A_local, local_rows * n,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Allocate a buffer for the local factorisation
  double *local_factor = malloc(local_rows * n * sizeof(double));
  if (!local_factor) {
    fprintf(stderr, "Memory allocation error on local_factor\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Copy the local data into the working buffer
  for (int i = 0; i < local_rows * n; i++) {
    local_factor[i] = A_local[i];
  }

  // Allocate memory for the Householder reflector coefficients
  double *tau = malloc(n * sizeof(double));
  if (!tau) {
    fprintf(stderr, "Memory allocation error on tau\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Perform the local QR factorisation
  int info =
      LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, local_rows, n, local_factor, n, tau);
  if (info != 0) {
    fprintf(stderr, "Error in LAPACKE_dgeqrf on rank %d; info = %d\n", rank,
            info);
    MPI_Abort(MPI_COMM_WORLD, info);
  }

  // Allocate memory to store the local R factor
  double *R_local = malloc(n * n * sizeof(double));
  if (!R_local) {
    fprintf(stderr, "Memory allocation error on R_local\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Extract the upper triangular part
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (j >= i)
        R_local[i * n + j] = local_factor[i * n + j];
      else
        R_local[i * n + j] = 0.0;
    }
  }

  // Generate explicit local Q factor
  info =
      LAPACKE_dorgqr(LAPACK_ROW_MAJOR, local_rows, n, n, local_factor, n, tau);
  if (info != 0) {
    fprintf(stderr, "Error in LAPACKE_dorgqr on rank %d; info = %d\n", rank,
            info);
    MPI_Abort(MPI_COMM_WORLD, info);
  }

  // Now contains the local Q factor
  double *Q_local = local_factor;

  // No longer needed
  free(tau);

  // Allocate memory on the root process for the stacked R matrix
  double *R_stack = NULL;
  if (rank == 0) {
    R_stack = malloc(size * n * n * sizeof(double));
    if (!R_stack) {
      fprintf(stderr, "Memory allocation error on R_stack\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Gather the local R factors from each factor into the stacked R matrix
  MPI_Gather(R_local, n * n, MPI_DOUBLE, R_stack, n * n, MPI_DOUBLE, 0,
             MPI_COMM_WORLD);

  // Free, since it is now in the stacked R matrix
  free(R_local);

  // Allocate memory for the final R factor
  double *R_final = malloc(n * n * sizeof(double));

  // This will hold the Q factor from the global QR factorisation on the stacked
  // R matrix
  double *Q_red_all = NULL;
  if (rank == 0) {
    int rows_R = size * n; // Total rows in the stacked R matrix

    // Allocate memory
    double *tau_red = malloc(n * sizeof(double));
    if (!tau_red) {
      fprintf(stderr, "Memory allocation error on tau_red\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // QR factorisation on the stacked R matrix to compute the final R
    info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, rows_R, n, R_stack, n, tau_red);
    if (info != 0) {
      fprintf(stderr, "Error in LAPACKE_dgeqrf on R_stack; info = %d\n", info);
      MPI_Abort(MPI_COMM_WORLD, info);
    }

    // Extract final R
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        if (j >= i)
          R_final[i * n + j] = R_stack[i * n + j];
        else
          R_final[i * n + j] = 0.0;
      }
    }

    // Compute the Q factor
    info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, rows_R, n, n, R_stack, n, tau_red);
    if (info != 0) {
      fprintf(stderr, "Error in LAPACKE_dorgqr on R_stack; info = %d\n", info);
      MPI_Abort(MPI_COMM_WORLD, info);
    }

    // Allocate memory for the intermediate Q matrix obtained from the stacked R
    // matrix
    Q_red_all = malloc(rows_R * n * sizeof(double));
    if (!Q_red_all) {
      fprintf(stderr, "Memory allocation error on Q_red_all\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Store Q obtained from stacked R matrix in this intermediate matrix
    for (int i = 0; i < rows_R * n; i++) {
      Q_red_all[i] = R_stack[i];
    }

    // No longer needed
    free(tau_red);
    free(R_stack);
  }

  // Size of this intermediate matrix
  int Q_red_size = size * n * n;

  // Processes which are not the root one must allocate memory to receive their
  // respective part of the intermediate matrix
  if (rank != 0) {
    Q_red_all = malloc(Q_red_size * sizeof(double));
    if (!Q_red_all) {
      fprintf(stderr, "Memory allocation error on Q_red_all\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }

  // Broadcast from root to all processes
  MPI_Bcast(Q_red_all, Q_red_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Each process extracts its corresponding block
  double Q_red_block[n * n];
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      Q_red_block[i * n + j] = Q_red_all[(rank * n + i) * n + j];
    }
  }

  // No longer needed since every process has its corresponding part
  free(Q_red_all);

  // Multiply local Q factors with the corresponding block in the intermediate Q
  // matrix to obtain the local orthogonal matrix
  double *Q_local_final = malloc(local_rows * n * sizeof(double));
  if (!Q_local_final) {
    fprintf(stderr, "Memory allocation error on Q_local_final.\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, local_rows, n, n, 1.0,
              Q_local, n, Q_red_block, n, 0.0, Q_local_final, n);

  // Original local block is no longer needed
  free(A_local);

  // Gather all updated local Q factors from all processes to form the final Q
  // matrix
  double *Q_final_global = NULL;
  if (rank == 0) {
    Q_final_global = malloc(m * n * sizeof(double));
    if (!Q_final_global) {
      fprintf(stderr, "Memory allocation error on Q_final_global\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
  }
  MPI_Gather(Q_local_final, local_rows * n, MPI_DOUBLE, Q_final_global,
             local_rows * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // No longer needed since it has been gathered
  free(Q_local_final);

  // This is where we verify the accuracy of the factorisation
  if (rank == 0) {

    // Allocate memory for the reconstructed matrix
    double *A_reconstructed = malloc(m * n * sizeof(double));
    if (!A_reconstructed) {
      fprintf(stderr, "Memory allocation error on A_reconstructed\n");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Reconstruct matrix using the final Q and R factors
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n, 1.0,
                Q_final_global, n, R_final, n, 0.0, A_reconstructed, n);

    // Compute the residual norm
    double error_norm = 0.0;
    for (int i = 0; i < m * n; i++) {
      double diff = A_global[i] - A_reconstructed[i];
      error_norm += diff * diff;
    }
    error_norm = sqrt(error_norm);

    // Print
    print_matrix("Original matrix, A", A_global, m, n);
    print_matrix("Final orthogonal matrix, Q", Q_final_global, m, n);
    print_matrix("Final upper triangular matrix, R", R_final, n, n);
    print_matrix("Reconstructed matrix, QR", A_reconstructed, m, n);
    printf("Residual norm, ||A - QR|| = %.16e\n", error_norm);

    // Free remaining memory
    free(A_reconstructed);
    free(A_global);
    free(Q_final_global);
    free(R_final);
  }
}

/**
 * @brief Main function.
 *
 * Initialises the MPI environment, performs the TSQR factorisation, and then
 * finalises the MPI environment.
 *
 * @returns 0 upon successful execution.
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  tsqr(16, 4);
  MPI_Finalize();
  return 0;
}
