/**
 * @file vector_copy.cc
 * @brief Code for 5614 L5 example
 * @author R. Morrin
 * @version 4.0
 * @date 2023-02-09
 */
#include <iostream>

namespace HPC
{
    class Vector
    {
	public:
	    Vector(const Vector & old); // Copy constructor

	    // Constructor
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

	    double& operator[](int i) {
		if(i >= number_of_elements){
		}
		return data[i]; 
	    }

	    // Could also just use operator[] top change values
	    void update_idx(int idx, double val){ data[idx] = val; }

	private:
	    int number_of_elements;
	    double * data;
    };
} /* HPC */ 

// Copy Constructor definition
HPC::Vector::Vector(const Vector & old){
    std::cout << "Creating copy" << std::endl;
    number_of_elements = old.number_of_elements;
    data = new double [number_of_elements];
    for (auto i = 0; i < number_of_elements; ++i) {
	data [i] = old.data[i];
    }
}

int main()
{
    HPC::Vector A {3}; 
    HPC::Vector B {A}; 	//Initialise B with A

    for (auto i = 0; i < 3; ++i) {
	A.update_idx(i,5.0); 	// Update A values to hold "5"
	std::cout << "B[" << i << "]="<< B[i] << std::endl; 
    }
    return 0;
}
