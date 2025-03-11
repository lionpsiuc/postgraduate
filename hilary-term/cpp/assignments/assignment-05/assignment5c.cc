#include <cmath>
#include <execution>
#include <iostream>
#include <vector>

struct transform_op {
  double operator()(double x) { return (x * std::log(x)); }
};

int main(void) {
  std::vector<double> A(10000);
  std::iota(A.begin(), A.end(), 1.0);

  /// TODO
  /// Write a lambda function you can use instead of transform_op
  ///
  std::transform(std::execution::par, A.begin(), A.end(), A.begin(),
                 transform_op{});

  for (auto const &a : A) {
    std::cout << a << ' ';
  }
  std::cout << '\n';

  return 0;
}
