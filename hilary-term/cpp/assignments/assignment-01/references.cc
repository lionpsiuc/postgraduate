/**
 * @file references.cc
 *
 * @brief Demonstrates the use of passing by reference.
 *
 * This programme defines a Vec struct to represent mathematical vectors and
 * implements functions to compute the dot product, normalise vectors, and sum
 * two vectors.
 *
 * @author Ion Lipsiuc
 * @date 2025-02-11
 * @version 1.0
 */

#include <cmath>
#include <iostream>

/**
 * @brief A struct to store a mathematical vector.
 *
 * This struct contains a pointer to an array of doubles and an integer storing
 * the number of elements in that array. It also includes a member function to
 * print the vector elements.
 */
struct Vec {
  double *values;   // Pointer to the array of vector elements
  int num_elements; // Number of elements in the vector

  /**
   * @brief Prints the elements of the vector.
   */
  void print(void) {
    if (num_elements > 0) {
      std::cout << "{";
      for (int i = 0; i < num_elements - 1; i++) {
        std::cout << values[i] << ", ";
      }
      std::cout << values[num_elements - 1] << "}";
    } else {
      std::cout << "Empty Vec\n";
    }
  }
};

/**
 * @brief Computes the dot product of two vectors.
 *
 * @param[in] u First input vector.
 * @param[in] v Second input vector.
 *
 * @returns The dot product of vectors u and v.
 */
double dot_prod(Vec const &u, Vec const &v) {
  if (u.num_elements != v.num_elements) {
    std::cerr << "Error: Size mismatch\n";
    std::exit(EXIT_FAILURE);
  }
  double result{0};
  for (int i = 0; i < u.num_elements; ++i) {
    result += u.values[i] * v.values[i];
  }
  return result;
}

/**
 * @brief Normalises a vector to unit length.
 *
 * @param[in,out] u The vector to be normalised.
 */
void normalise(Vec &u) {
  double len{0};
  for (int i = 0; i < u.num_elements; ++i) {
    len += u.values[i] * u.values[i];
  }
  len = std::sqrt(len);
  for (int i = 0; i < u.num_elements; ++i) {
    u.values[i] /= len;
  }
}

/**
 * @brief Computes the sum of two vectors.
 *
 * @param[in] u First input vector.
 * @param[in] v Second input vector.
 *
 * @returns A reference to a dynamically allocated vector containing the sum of
 * u and v.
 */
Vec &sum(Vec const &u, Vec const &v) {
  if (u.num_elements != v.num_elements) {
    std::cerr << "Error: Size mismatch\n";
    std::exit(EXIT_FAILURE);
  }

  // Allocate new vector to store result
  Vec *result = new Vec;
  result->num_elements = u.num_elements;
  result->values = new double[result->num_elements];

  // Compute sum of each corresponding element
  for (int i = 0; i < u.num_elements; ++i) {
    result->values[i] = u.values[i] + v.values[i];
  }

  return *result;
}

/**
 * @brief Main function.
 *
 * Demonstrates vector operations. The programme initialises two vectors, prints
 * them, computes their dot product, sums them, normalises one of them, and then
 * prints the results.
 *
 * @returns 0 upon successful execution.
 */
int main(void) {

  // Create first vector
  Vec v1;
  v1.num_elements = 10;
  v1.values = new double[v1.num_elements];
  for (int i = 0; i < v1.num_elements; ++i) {
    v1.values[i] = i;
  }

  // Create second vector
  Vec v2;
  v2.num_elements = v1.num_elements;
  v2.values = new double[v2.num_elements];
  for (int i = 0; i < v2.num_elements; ++i) {
    v2.values[v2.num_elements - 1 - i] = i / 2.0;
  }

  // Print initial vectors
  std::cout << "v1:          ";
  v1.print();
  std::cout << "\nv2:          ";
  v2.print();
  std::cout << "\n";

  // Compute and print the dot product
  std::cout << "Dot product: " << dot_prod(v1, v2) << "\n";

  // Compute and print the sum of vectors
  Vec &vsum = sum(v1, v2);
  std::cout << "Sum:         ";
  vsum.print();
  std::cout << "\n";

  // Normalise and print v1
  normalise(v1);
  std::cout << "v1 norm:     ";
  v1.print();
  std::cout << "\n";

  // Free dynamically allocated memory
  delete[] v1.values;
  delete[] v2.values;
  delete[] vsum.values;
  delete &vsum;

  return 0;
}
