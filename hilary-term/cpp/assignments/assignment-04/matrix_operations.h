/**
 * @file matrix_operations.h
 * @brief Contains function declaration for a function to perform Gauss-Jordan
 * elimination.
 * @author R. Morrin
 * @version 3.0
 * @date 2025-02-25
 */
#ifndef MATRIX_OPERATIONS_H_A4QKSOWJ
#define MATRIX_OPERATIONS_H_A4QKSOWJ

#include "hpc_concepts.h"
#include "matrix.h"

template <Number T, Number U>
HPC::Matrix<double> gaussjordan(HPC::Matrix<T> const &A,
                                HPC::Matrix<U> const &B);

#endif /* end of include guard: MATRIX_OPERATIONS_H_A4QKSOWJ */
