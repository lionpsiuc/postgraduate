/**
 * @file is_floating.cc
 * @brief  Toy example to show how you might implement a class/struct to provide
 * some type trait information
 * @author R. Morrin
 * @version 1.0
 * @date 2024-03-09
 */

#include <iomanip>
#include <iostream>

template <typename T>
struct is_floating {
  static const bool value{false};
};

/// TODO
/// Specialise the above struct for doubles and floats

/// END of changes

int main() {
  auto a{1.0};
  auto b{2};
  auto c{3.4f};

  std::cout << std::boolalpha // Print as "true" and "false" (0)
            << "a:\t" << is_floating<decltype(a)>::value << '\n'
            << "b:\t" << is_floating<decltype(b)>::value << '\n'
            << "c:\t" << is_floating<decltype(c)>::value << '\n';

  return 0;
}
