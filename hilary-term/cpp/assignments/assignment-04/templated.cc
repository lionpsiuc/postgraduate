/**
 * @file templated.cc
 * @brief Toy example to show the importance of templates.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#include <cstddef>
#include <iostream>

/**
 * @brief Templated function to compute the dot product of two vectors.
 *
 * @param[in] x Pointer to the first array of type T1
 * @param[in] y Pointer to second array of type T2
 * @param[in] N Number of points in each array
 *
 * @returns The dot product type determined by the product of T1 and T2.
 */
template <typename T1, typename T2>
auto dot(const T1 *x, const T2 *y, int N) {

  // Initialise result with correct type based on multiplication of T1 and T2
  auto result = T1{0} * T2{0};

  for (int i = 0; i < N; ++i) {
    result += x[i] * y[i];
  }
  return result;
}

/**
 * @brief Main function.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  int n{4};
  double A[]{1, 2, 3, 4};
  int B[]{5, 6, 7, 8};
  std::cout << "A.A = " << dot(A, A, n) << "\nB.B = " << dot(B, B, n)
            << "\nA.B = " << dot(A, B, n) << "\nB.A = " << dot(B, A, n) << "\n";
  return 0;
}
