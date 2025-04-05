/**
 * @file is-floating.cc
 * @brief Toy example we had to modify to ensure that the struct is specialised
 *        for doubles and floats.
 *
 * @author Ion Lipsiuc
 * @version 1.0
 * @date 2024-03-30
 */

#include <iomanip>
#include <iostream>

/**
 * @brief Main template for checking if a type is a float.
 *
 * This is the general case which defaults to false for all types.
 *
 * @tparam T The type to check.
 */

template <typename T>
struct is_floating {
  static const bool value{false};
};

/**
 * @brief Specialisation for doubles.
 */
template <>
struct is_floating<double> {
  static const bool value{true};
};

/**
 * @brief Specialisation for floats.
 */
template <>
struct is_floating<float> {
  static const bool value{true};
};

/**
 * @brief Main function.
 *
 * Carries out matrix and vector operations.
 *
 * @returns 0 upon successful execution.
 */
int main() {
  auto a{1.0};
  auto b{2};
  auto c{3.4f};
  std::cout << std::boolalpha << "a:\t" << is_floating<decltype(a)>::value
            << "\n"
            << "b:\t" << is_floating<decltype(b)>::value << "\n"
            << "c:\t" << is_floating<decltype(c)>::value << "\n";
  return 0;
}
