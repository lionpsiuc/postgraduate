/**
 * @file ctad.cc
 * @brief Toy example we had to modify to use class template argument deduction
 *        (CTAD).
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2025-03-30
 */

#include <iostream>
#include <vector>

/**
 * @brief Main function.
 *
 * Toy example I modified to include CTAD.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  std::vector A{1.0, 2, 3};
  for (auto &i : A) {
    i /= 2;
  }
  for (auto const &i : A) {
    std::cout << i << "\n";
  }
  return 0;
}
