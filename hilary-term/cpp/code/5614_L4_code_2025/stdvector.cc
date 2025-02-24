/**
 * @file stdvector.cc
 * @brief Example to show some std::vector properties. 
 * 	Also, compile this with -D_GLIBCXX_ASSERTIONS 
 * 	show how gcc *can* implement bounds checking for 
 * 	std::vector.
 *
 * @author R. Morrin
 * @version 1.0
 * @date 2023-02-09
 */
#include <vector>
#include <iostream>

int main()
{
  std::vector A {3,4,5,6}; 

  // capacity might not be the same as size. size is
  // the number of elements. capacity is the number 
  // that can be held in currently allocate storage
  std::cout << "A.size() = " << A.size() 
   << "\nA.capacity() = " << A.capacity() << "\n\n";

  A.erase(std::begin(A)); // Erase first element
  std::cout << "A.size() = " << A.size() 
   << "\nA.capacity() = " << A.capacity() << "\n\n";

  for (auto i = 0; i < 5; ++i) {
    std::cout << "A[" << i << "]="<< A[i] << '\n';
  }
  return 0;
}
