#include <iostream>
#include <array>
#include <algorithm>

struct Greater_than {
  bool operator()(int a, int b){
    return a>b;
  }
};

int main()
{
  std::array<int, 5> arr {4, 2, 1, 0, 3}; 
  Greater_than comp {};

  std::cout << "Original array\n";
  for (auto i: arr) {
   std::cout << i << ' ';
  }
  
  // Sort array and print
  std::sort(std::begin(arr), std::end(arr));
  std::cout << "\n\nAfter sorting\n";
  for (auto i: arr) {
   std::cout << i << ' ';
  }

  // Sort in reverse using functor to compare
  std::sort(std::begin(arr), std::end(arr), comp);
  std::cout << "\n\nAfter sorting reverse\n";
  for (auto i: arr) {
   std::cout << i << ' ';
  }
  return 0;
}
