#include <array>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>

int main()
{
  std::array<double, 5> arr {3.1, 2.2, 6.3, 8.4, -9.1};

  std::cout << "Accumulate = " << std::accumulate(arr.begin(), arr.end(),0) << '\n';
  std::cout << "Accumulate2 = " << std::accumulate(arr.begin(), arr.end(),0.0) << '\n';

  std::cout << "Reduce = " << std::reduce(arr.begin(), arr.end(),0) << '\n';
  std::cout << "Reduce2 = " << std::reduce(arr.begin(), arr.end(),0.0) << '\n';
  return 0;
}
