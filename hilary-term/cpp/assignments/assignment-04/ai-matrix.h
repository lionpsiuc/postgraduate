#ifndef AI_MATRIX_H
#define AI_MATRIX_H

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
class Matrix {
private:
  size_t rows, cols;
  std::vector<T> data;

public:
  // Default constructor
  Matrix() : rows(0), cols(0) {}

  // Constructor with dimensions and an optional default value
  Matrix(size_t r, size_t c, const T &value = T())
      : rows(r), cols(c), data(r * c, value) {}

  // Initialiser list constructor
  Matrix(std::initializer_list<std::initializer_list<T>> init) {
    rows = init.size();
    cols = (rows > 0) ? init.begin()->size() : 0;
    data.reserve(rows * cols);
    for (const auto &row : init) {
      if (row.size() != cols) {
        throw std::invalid_argument(
            "All rows must have the same number of columns");
      }
      data.insert(data.end(), row.begin(), row.end());
    }
  }

  // Copy constructor
  Matrix(const Matrix &other)
      : rows(other.rows), cols(other.cols), data(other.data) {
    std::cout << "Copy constructor called" << std::endl;
  }

  // Move constructor
  Matrix(Matrix &&other) noexcept
      : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
    other.rows = 0;
    other.cols = 0;
  }

  // Copy assignment operator
  Matrix &operator=(const Matrix &other) {
    if (this != &other) {
      rows = other.rows;
      cols = other.cols;
      data = other.data;
    }
    return *this;
  }

  // Move assignment operator
  Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      rows = other.rows;
      cols = other.cols;
      data = std::move(other.data);
      other.rows = 0;
      other.cols = 0;
    }
    return *this;
  }

  // Element access operator (non-const version)
  T &operator()(size_t i, size_t j) {
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Matrix indices out of range");
    }
    return data[i * cols + j];
  }

  // Element access operator (const version)
  const T &operator()(size_t i, size_t j) const {
    if (i >= rows || j >= cols) {
      throw std::out_of_range("Matrix indices out of range");
    }
    return data[i * cols + j];
  }

  // Getters for dimensions
  size_t numRows() const { return rows; }
  size_t numCols() const { return cols; }

  // A simple print method for debugging
  void print() const {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < cols; ++j) {
        std::cout << (*this)(i, j) << " ";
      }
      std::cout << std::endl;
    }
  }
};

#endif
