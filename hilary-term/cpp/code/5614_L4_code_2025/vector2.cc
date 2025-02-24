/**
 * @file vector2.cc
 * @brief Code for 5614 L5 example
 * @author R. Morrin
 * @version 5.0
 * @date 2024-02-09
 */
#include <iostream>

namespace HPC {
  class Vector {
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
      }

      double& operator[](int i) {
	if(i >= number_of_elements || i < 0){
	  std::cerr << "ERROR: Accesing beyond bounds!!!!" << std::endl;
	}
	return data[i]; 
      }

    private:
      int number_of_elements;
      double * data;
  };
} /* HPC */ 

int main()
{
  HPC::Vector A {3}; 
  // HPC::Vector B;

  for (auto i = 0; i < 5; ++i) {
    std::cout << "A[" << i << "]="<< A[i] << std::endl; 
  }
  return 0;
}
