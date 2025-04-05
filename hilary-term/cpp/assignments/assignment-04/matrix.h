/**
 * @file matrix.h
 * @brief Templated class for a matrix.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#ifndef MATRIX_H
#define MATRIX_H

#include "hpc-concepts.h"
#include "logging.h"

#include <fstream>
#include <iomanip>
#include <iostream>

namespace HPC {

/**
 * @brief A templated class for a matrix allowing numerical operations.
 *
 * @tparam T The data type stored in the matrix.
 */
template <Number T>
class Matrix {
public:
  /**
   * @brief Default constructor is deleted.
   */
  Matrix() = delete;

  /**
   * @brief Constructs a matrix with specified dimensions.
   *
   * @param rows Number of rows in the matrix
   * @param cols Number of columns in the matrix
   */
  Matrix(std::size_t const rows, std::size_t const cols);

  /**
   * @brief Constructs a matrix by reading data from a file.
   *
   * @param file Path to the file containing matrix data.
   */
  Matrix(std::string const file);

  /**
   * @brief Access operator for modifying matrix elements.
   *
   * @param row_num Row index.
   * @param col_num Column index.
   *
   * @returns Reference to the element at specified position.
   */
  T &operator()(std::size_t const row_num, std::size_t const col_num);

  /**
   * @brief Access operator for reading matrix elements.
   *
   * Modified to be a const access operator as per the assignment instructions.
   *
   * @param row_num Row index.
   * @param col_num Column index.
   *
   * @returns Copy of the element at specified position
   */
  T operator()(std::size_t const row_num, std::size_t const col_num) const;

  /**
   * @brief Destructor that releases allocated memory.
   */
  ~Matrix();

  /**
   * @brief Copy constructor.
   *
   * @param m Matrix to copy from
   */
  Matrix(Matrix const &m);

  /**
   * @brief Copy assignment operator.
   *
   * @param m Matrix to copy from.
   *
   * @returns Reference to this matrix after assignment.
   */
  Matrix &operator=(Matrix const &m);

  /**
   * @brief Move constructor.
   *
   * @param in Matrix to move resources from.
   */
  Matrix(Matrix &&in);

  /**
   * @brief Move assignment operator.
   *
   * @param in Matrix to move resources from.
   *
   * @returns Reference to this matrix after assignment
   */
  Matrix &operator=(Matrix &&in);

  /**
   * @brief Stream output operator for printing matrices.
   *
   * @param os Output stream.
   * @param in Matrix to output.
   *
   * @returns Modified output stream.
   */
  friend std::ostream &operator<< <>(std::ostream &os, Matrix<T> const &in);

  /**
   * @brief Gets the number of rows in the matrix.
   *
   * @returns Number of rows.
   */
  std::size_t get_num_rows() const { return rows; };

  /**
   * @brief Gets the number of columns in the matrix.
   *
   * @returns Number of columns.
   */
  std::size_t get_num_cols() const { return cols; };

private:
  std::size_t rows, cols; // Dimensions of the matrix
  T *data;                // Pointer to dynamically allocated data.
};

/**
 * @brief Constructor that initialises all elements to zero.
 */
template <Number T>
Matrix<T>::Matrix(std::size_t const num_rows, std::size_t const num_cols)
    : rows{num_rows}, cols{num_cols}, data{new T[rows * cols]{}} {
  std::cout << "Constructing a matrix of size " << rows << " x " << cols
            << "\n";
}

/**
 * @brief Constructor that reads matrix data from a file.
 */
template <Number T>
Matrix<T>::Matrix(std::string const file) : rows{0}, cols{0}, data{nullptr} {
  std::ifstream infile(file);
  if (!infile.is_open()) {
    std::cerr << "Could not open file " << file << std::endl;
    throw std::runtime_error("Could not open file: " + file);
  }

  // Read the number of rows and columns
  infile >> rows >> cols;

  // Allocate memory
  data = new T[rows * cols];

  // Read the data
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (!(infile >> data[i * cols + j])) {
        delete[] data;
        throw std::runtime_error("Could not read data from file: " + file);
      }
    }
  }

  std::cout << "Constructing a matrix with the data read from file " << file
            << std::endl;
}

/**
 * @brief Destructor that frees allocated memory.
 */
template <Number T>
Matrix<T>::~Matrix() {
  delete[] data;
  std::cout << "Calling destructor for a matrix of size " << rows << " x "
            << cols << std::endl;
}

/**
 * @brief Copy constructor.
 */
template <Number T>
Matrix<T>::Matrix(Matrix const &m)
    : rows{m.rows}, cols{m.cols}, data{new T[rows * cols]} {
  std::copy(m.data, m.data + (rows * cols), data);
  std::cout << "Copy constructor" << std::endl;
}

/**
 * @brief Copy assignment operator.
 */
template <Number T>
Matrix<T> &Matrix<T>::operator=(Matrix const &m) {
  std::cout << "Copy assignment operator" << std::endl;

  // Check for self-assignment
  if (this == &m) {
    return *this;
  }

  // If dimensions are different, need to reallocate
  if (rows != m.rows || cols != m.cols) {
    delete[] data;
    rows = m.rows;
    cols = m.cols;
    data = new T[rows * cols];
  }

  std::copy(m.data, m.data + (rows * cols), data);
  return *this;
}

/**
 * @brief Move constructor.
 */
template <Number T>
Matrix<T>::Matrix(Matrix &&m) : rows{m.rows}, cols{m.cols}, data{m.data} {
  std::cout << "Move constructor" << std::endl;
  m.data = nullptr;
  m.rows = 0;
  m.cols = 0;
}

/**
 * @brief Move assignment operator.
 */
template <Number T>
Matrix<T> &Matrix<T>::operator=(Matrix &&m) {
  std::cout << "Move assignment operator" << std::endl;

  // Check for self-assignment
  if (this == &m) {
    return *this;
  }

  // Free current resources
  delete[] data;

  rows = m.rows;
  cols = m.cols;
  data = m.data;

  // Reset the source object
  m.data = nullptr;
  m.rows = 0;
  m.cols = 0;

  return *this;
}

/**
 * @brief Access operator, modified for non-const.
 */
template <Number T>
T &Matrix<T>::operator()(std::size_t const row_num, std::size_t const col_num) {
  if (row_num >= rows || col_num >= cols) {
    std::cerr << "Matrix of size " << rows << " x " << cols << " at subscript ("
              << row_num << ", " << col_num
              << ") is out of bounds: " << sourceline() << std::endl;
    throw std::out_of_range("Matrix indices out of bounds");
  }

  // Return reference to the element
  return data[row_num * cols + col_num];
}

/**
 * @brief Access operator, modified for const.
 */
template <Number T>
T Matrix<T>::operator()(std::size_t const row_num,
                        std::size_t const col_num) const {

  // Check if indices are within bounds
  if (row_num >= rows || col_num >= cols) {
    std::cerr << "Matrix of size " << rows << " x " << cols << " at subscript ("
              << row_num << ", " << col_num
              << ") is out of bounds: " << sourceline() << std::endl;
    throw std::out_of_range("Matrix indices out of bounds");
  }

  // Return copy of the element
  return data[row_num * cols + col_num];
}

/**
 * @brief Stream insertion operator for output from a matrix.
 *
 * @tparam T Element type of the matrix.
 *
 * @param[in,out] os Output stream.
 * @param[in] in Matrix to output.
 *
 * @returns Modified output stream.
 */
template <Number T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &in) {
  for (std::size_t i = 0; i < in.get_num_rows(); ++i) {
    os << "| ";
    for (std::size_t j = 0; j < in.get_num_cols(); ++j) {
      os << std::setw(6) << std::fixed << std::setprecision(2) << in(i, j)
         << " ";
    }
    os << "|\n";
  }
  return os;
}

/**
 * @brief Matrix addition operator.
 *
 * @tparam T1 Element type of first matrix.
 * @tparam T2 Element type of second matrix.
 *
 * @param[in] M1 First matrix.
 * @param[in] M2 Second matrix.
 *
 * @returns A matrix with its element type deduced from T1 and T2.
 */
template <Number T1, Number T2>
auto operator+(HPC::Matrix<T1> const &M1, HPC::Matrix<T2> const &M2) {

  // Check if matrices have the same dimensions
  if (M1.get_num_rows() != M2.get_num_rows() ||
      M1.get_num_cols() != M2.get_num_cols()) {
    std::cerr << "Incompatible matrix dimensions for addition: " << sourceline()
              << std::endl;
    throw std::invalid_argument("Incompatible matrix dimensions for addition");
  }

  // Check what the return type should be
  using ReturnType = decltype(T1{} + T2{});

  // Create result matrix
  HPC::Matrix<ReturnType> result(M1.get_num_rows(), M1.get_num_cols());

  // Perform addition
  for (std::size_t i = 0; i < M1.get_num_rows(); ++i) {
    for (std::size_t j = 0; j < M1.get_num_cols(); ++j) {
      result(i, j) = M1(i, j) + M2(i, j);
    }
  }

  return result;
}

/**
 * @brief Matrix multiplication operator.
 *
 * @tparam T1 Element type of first matrix.
 * @tparam T2 Element type of second matrix.
 *
 * @param[in] M1 First matrix.
 * @param[in] M2 Second matrix.
 *
 * @returns A matrix with its element type deduced from T1 and T2.
 */
template <Number T1, Number T2>
auto operator*(HPC::Matrix<T1> const &M1, HPC::Matrix<T2> const &M2) {

  // Check if matrices have compatible dimensions for multiplication
  if (M1.get_num_cols() != M2.get_num_rows()) {
    std::cerr << "Incompatible matrix dimensions for multiplication: "
              << sourceline() << std::endl;
    throw std::invalid_argument(
        "Incompatible matrix dimensions for multiplication");
  }

  // Check what the return type should be
  using ReturnType = decltype(T1{} * T2{});

  // Create result matrix
  HPC::Matrix<ReturnType> result(M1.get_num_rows(), M2.get_num_cols());

  // Perform multiplication
  for (std::size_t i = 0; i < M1.get_num_rows(); ++i) {
    for (std::size_t j = 0; j < M2.get_num_cols(); ++j) {
      ReturnType sum{};
      for (std::size_t k = 0; k < M1.get_num_cols(); ++k) {
        sum += M1(i, k) * M2(k, j);
      }
      result(i, j) = sum;
    }
  }

  return result;
}

/**
 * @brief Converts a matrix into an identity matrix.
 *
 * @param[in] in Matrix to convert to an identity matrix.
 */
void identity(auto &in) {

  // Check if the matrix is square
  if (in.get_num_rows() != in.get_num_cols()) {
    std::cerr << "Identity matrix must be square: " << sourceline()
              << std::endl;
    throw std::invalid_argument("Identity matrix must be square");
  }

  // Set all elements to zero
  for (std::size_t i = 0; i < in.get_num_rows(); ++i) {
    for (std::size_t j = 0; j < in.get_num_cols(); ++j) {
      in(i, j) = 0;
    }
  }

  // Set diagonal elements to one
  for (std::size_t i = 0; i < in.get_num_rows(); ++i) {
    in(i, i) = 1;
  }
}

} // namespace HPC

#endif
