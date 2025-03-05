#include <array>
#include <iostream>
#include <algorithm>
#include <numeric>

struct Positive {
  bool operator()(const int in){
	return in>=0;
  }
};

int main()
{
  std::array<int, 5> arr1 {3, 2, 6, 8, 9};
  std::array<int, 5> arr2 {3, 2, 6, 8, -9};
  Positive comp;

  std::cout << "Minimum arr1 is " << *(std::min_element(std::cbegin(arr1), std::cend(arr1))) << '\n'
  << "Minimum arr2 is " << *(std::min_element(std::cbegin(arr2), std::cend(arr2))) << '\n'
  << "Maximum arr1 is " << *(std::max_element(std::cbegin(arr1), std::cend(arr1))) << '\n'
  << "Maximum arr2 is " << *(std::max_element(std::cbegin(arr2), std::cend(arr2))) << '\n';
  std::cout << "Sum of all elements = " << std::accumulate(arr1.begin(), arr1.end(),0) << '\n';
  std::cout << "Sum of all elements2 = " << std::reduce(arr2.begin(), arr2.end(),0) << '\n';
  // Positive {} create an anonyous object of type struct Positive
  // This works the same as using "comp"
  if(std::all_of(std::cbegin(arr1), std::cend(arr1), Positive {})){ 	// Ok
	  std::cout << "Arr1:All positive\n";
  }
  if(std::all_of(std::cbegin(arr2), std::cend(arr2), comp)){ 		// Should return false
	  std::cout << "Arr2: All positive\n";
  }
  return 0;
}
