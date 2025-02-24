/**
 * @file vector1.cc
 * @brief Code for 5614 L4 example
 * @author R. Morrin
 * @version 5.0
 * @date 2024-02-09
 */
#include <iostream>

namespace HPC
{
  class Vector
  {
    public:
      Vector() = delete; 

      Vector (int num) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements << "\n";
	for (auto i = 0; i < num; ++i) {
	  data[i]=0; 
	}
      };

      ~Vector (){
	std::cout << "Destroying vector of size " << number_of_elements << "\n";
	delete[] data;
      };

      double get_element_by_idx(int idx){
	return data[idx];
      }

    private:
      int number_of_elements;
      double * data;
  };

} /* HPC */ 

int main()
{
  HPC::Vector A {3}; 
//  HPC::Vector B;  // Error. Cannot access the default conctructor

  for (auto i = 0; i < 5; ++i) {
    std::cout << "A[" << i << "]="<< A.get_element_by_idx(i) << std::endl; 
  }

  return 0;
}
