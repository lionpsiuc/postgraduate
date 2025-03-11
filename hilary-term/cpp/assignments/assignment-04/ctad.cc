/**
 * @file ctad.cc
 * @brief Written question for Assignment 4. 5614.
 * 	Either include the line you need to change in you written document, or
 * modify this file and upload it. The rest of the code is just so that you can
 * see whether your answer works.
 * @author R. Morrin
 * @version 1.0
 * @date 2025-02-23
 */

#include <iostream>
#include <vector>

int main() {
  std::vector<double> A{1, 2, 3};

  for (auto &i : A) {
    i /= 2;
  }

  for (auto const &i : A) {
    std::cout << i << '\n';
  }

  return 0;
}
