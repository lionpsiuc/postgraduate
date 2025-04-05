/**
 * @file matrix-operations.cc
 * @brief Implementation of a Gauss-Jordan elimination function.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#include "matrix-operations.h"
#include "logging.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

/**
 * @brief Performs Gauss-Jordan elimination on matrix A and applies the same
 *        operations to matrix B
 *
 * @tparam T Number type of matrix A
 * @tparam U Number type of matrix B
 *
 * @param[in] A Input matrix to be reduced to identity, which must be square.
 * @param[in] B Second matrix on which the same operations are performed.
 *
 * @returns The transformed matrix B.
 */
template <Number T, Number U>
HPC::Matrix<double> gaussjordan(HPC::Matrix<T> const &A,
                                HPC::Matrix<U> const &B) {

  // Check if A is square
  if (A.get_num_rows() != A.get_num_cols()) {
    std::cerr << "Matrix A must be square for Gauss-Jordan elimination: "
              << sourceline() << std::endl;
    throw std::invalid_argument("Matrix A must be square");
  }

  // Check if A and B have compatible dimensions
  if (A.get_num_rows() != B.get_num_rows()) {
    std::cerr << "Matrices A and B must have the same number of rows: "
              << sourceline() << std::endl;
    throw std::invalid_argument("Matrix dimensions do not match");
  }

  // Get dimensions
  std::size_t n = A.get_num_rows();
  std::size_t m = B.get_num_cols();

  // Create copies of A and B with double precision
  HPC::Matrix<double> A_copy(n, n);
  HPC::Matrix<double> B_copy(n, m);

  // Copy data from A and B to their respective copies
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      A_copy(i, j) = static_cast<double>(A(i, j));
    }
    for (std::size_t j = 0; j < m; ++j) {
      B_copy(i, j) = static_cast<double>(B(i, j));
    }
  }

  // Perform Gauss-Jordan elimination
  for (std::size_t i = 0; i < n; ++i) {

    // Find pivot
    double pivot = A_copy(i, i);

    // Normalise the current row
    for (std::size_t j = 0; j < n; ++j) {
      A_copy(i, j) /= pivot;
    }
    for (std::size_t j = 0; j < m; ++j) {
      B_copy(i, j) /= pivot;
    }

    // Eliminate other rows
    for (std::size_t k = 0; k < n; ++k) {
      if (k != i) {
        double factor = A_copy(k, i);

        // Subtract the scaled row from the current row
        for (std::size_t j = 0; j < n; ++j) {
          A_copy(k, j) -= factor * A_copy(i, j);
        }
        for (std::size_t j = 0; j < m; ++j) {
          B_copy(k, j) -= factor * B_copy(i, j);
        }
      }
    }
  }

  return B_copy;
}

// Explicit instantiation
template HPC::Matrix<double> gaussjordan<int, int>(HPC::Matrix<int> const &A,
                                                   HPC::Matrix<int> const &B);
template HPC::Matrix<double>
gaussjordan<double, double>(HPC::Matrix<double> const &A,
                            HPC::Matrix<double> const &B);
template HPC::Matrix<double>
gaussjordan<int, double>(HPC::Matrix<int> const &A,
                         HPC::Matrix<double> const &B);
template HPC::Matrix<double>
gaussjordan<double, int>(HPC::Matrix<double> const &A,
                         HPC::Matrix<int> const &B);
template HPC::Matrix<double>
gaussjordan<float, float>(HPC::Matrix<float> const &A,
                          HPC::Matrix<float> const &B);
template HPC::Matrix<double>
gaussjordan<float, double>(HPC::Matrix<float> const &A,
                           HPC::Matrix<double> const &B);
template HPC::Matrix<double>
gaussjordan<double, float>(HPC::Matrix<double> const &A,
                           HPC::Matrix<float> const &B);
template HPC::Matrix<double>
gaussjordan<int, float>(HPC::Matrix<int> const &A, HPC::Matrix<float> const &B);
template HPC::Matrix<double>
gaussjordan<float, int>(HPC::Matrix<float> const &A, HPC::Matrix<int> const &B);
