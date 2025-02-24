/**
 * @file vector4.cc
 * @brief Code for 5614 L5 example
 * @author R. Morrin
 * @version 4.0
 * @date 2023-02-10
 */
#include <iostream>

namespace HPC {
  class Vector {
    public:
      Vector (int num) : number_of_elements {num}, data {new double [num]}{
	for (auto i = 0; i < num; ++i) {
	  data[i]=0; 
	}
      };
      ~Vector (){
	//delete[] data;  	//NB removing this temporarily
      };

      double& operator[](int i) {
	if(i >= number_of_elements){
	}
	return data[i]; 
      }

      //void update_idx(int idx, double val){ data[idx] = val; }

    private:
      int number_of_elements;
      double * data;
  };
} /* HPC */ 

int main()
{
  HPC::Vector A {3}; 
  HPC::Vector B {A}; 	//Initialise B with A
  
  for (auto i = 0; i < 3; ++i) {
      //A.update_idx(i,5.0); 	// Update A values to hold "5"
      A[i] = 5.0;
      std::cout << "B[" << i << "]="<< B[i] << std::endl; 
  }
  return 0;
}
