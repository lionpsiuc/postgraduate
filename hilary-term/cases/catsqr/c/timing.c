/**
 * @file timing.c
 *
 * @brief Communication-avoiding TSQR factorisation using MPI and LAPACK with
 * timing functionality.
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
 * @brief Perform the TSQR factorisation on a distributed matrix.
 *
 * This function divides the global matrix among the available MPI processes,
 * computes a local QR factorisation on each process, gathers the local R
 * factors, and then performs a global QR factorisation on the stacked R matrix.
 * The final Q factor is obtained by updating the local Q factors using the
 * reduction Q factor computed from the stacked R matrix.
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

    // Fill the global matrix with random values
    for (int i = 0; i < m * n; i++) {
      A_global[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
  }

  // Scatter the global matrix to all processes
  MPI_Scatter(A_global, local_rows * n, MPI_DOUBLE, A_local, local_rows * n,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Free allocated memory
  if (rank == 0)
    free(A_global);

  // Allocate a buffer for the local factorisation
  double *local_factor = malloc(local_rows * n * sizeof(double));
  if (!local_factor) {
    fprintf(stderr, "Memory allocation error on local_factor\n");
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  // Copy the local data into the working buffer
  for (int i = 0; i < local_rows * n; i++)
    local_factor[i] = A_local[i];

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
    fprintf(stderr, "Memory allocation error on R_local.\n");
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
      fprintf(stderr, "Memory allocation error on R_stack.\n");
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
    for (int i = 0; i < rows_R * n; i++)
      Q_red_all[i] = R_stack[i];

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
    fprintf(stderr, "Memory allocation error on Q_local_final\n");
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

  // Free remaining memory
  if (rank == 0) {
    free(Q_final_global);
    free(R_final);
  }
}

/**
 * @brief Main function.
 *
 * Initialises the MPI environment, sets up a range of matrix dimensions, and
 * runs the TSQR factorisation for each combination of dimensions while
 * recording the execution time. The timing results are written to a file.
 *
 * @returns 0 upon successful execution.
 */
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Seed the random number generator
  srand(time(NULL) + rank);

  // Define an array for the number of rows and columns
  int n_values[] = {4, 8, 16, 32};
  int num_n = sizeof(n_values) / sizeof(int);
  int m_values[] = {1000, 10000, 100000, 500000, 1000000};
  int num_m = sizeof(m_values) / sizeof(int);

  // File pointer for outputting timing results
  FILE *fp = NULL;

  if (rank == 0) {
    fp = fopen("timing.txt", "w");
    if (!fp) {
      perror("Error opening timing.txt");
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Header for the output
    fprintf(fp, "n,m,time\n");
  }

  // Loop over each value for the number of columns
  for (int i = 0; i < num_n; i++) {
    int n = n_values[i];

    // Loop over each value for the number of rows
    for (int j = 0; j < num_m; j++) {
      int m = m_values[j];

      // Ensure that the number of rows is divisible by four
      if (m % 4 != 0)
        m = (m / 4) * 4;

      // Synchronise all processes before timing begins
      MPI_Barrier(MPI_COMM_WORLD);

      double t_start = MPI_Wtime();
      tsqr(m, n);

      // Synchronise to ensure all processes have completed the factorisation
      MPI_Barrier(MPI_COMM_WORLD);

      double t_end = MPI_Wtime();

      // Elapsed time
      double elapsed = t_end - t_start;

      // Root process writes results to the file
      if (rank == 0) {
        fprintf(fp, "%d,%d,%.6e\n", n, m, elapsed);
        fflush(fp);
      }
    }
  }

  // Root processes closes the output file
  if (rank == 0)
    fclose(fp);

  MPI_Finalize();
  return 0;
}
