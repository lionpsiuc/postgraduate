/**
 * @file matrix-operations.h
 * @brief Contains function declaration for a function to perform Gauss-Jordan
 *        elimination.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */
#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include "hpc-concepts.h"
#include "matrix.h"

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
                                HPC::Matrix<U> const &B);

#endif
