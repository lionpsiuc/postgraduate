#include <algorithm>
#include <iostream>
#include <vector>

/// TODO:
/// Write functor is_lower_than
/// to use instead of lambda below in the remove_if algorithm

int main(void) {
  std::vector V{-1.2, 5.4, 12.2, -3.9, -0.4, 13.1, 0.5};

  const double cutoff_value{1.1};

  auto lambda = [cutoff_value](double testval) {
    return testval < cutoff_value;
  };

  auto new_it = std::remove_if(std::begin(V), std::end(V), lambda);
  V.erase(new_it, std::end(V));

  // print filtered vector to screen
  for (auto const &v : V) {
    std::cout << v << " ";
  }
  std::cout << '\n';

  return 0;
}
