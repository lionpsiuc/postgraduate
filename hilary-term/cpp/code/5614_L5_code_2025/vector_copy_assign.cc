/**
 * @file vector_copy_assign.cc
 * @brief Code for 5614 L5 example
 * @author R. Morrin
 * @version 2.0
 * @date 2021-02-15
 */
#include <iostream>

namespace HPC
{
    class Vector
    {
	public:
	    Vector (int num); 			// Constructor declaration
	    Vector (int num, double init); 	// Overloaded onstructor declaration
	    Vector(const Vector & old); 	// Copy constructor declaration
	    ~Vector (); 			// Destructor declaration
	    Vector& operator=(const Vector &in);// Copy assignment operator 

	    double& operator[](int i) { 	//operator[] must be inlined
		if(i >= number_of_elements){
		    std::cerr << "ERROR: Accesing beyond bounds!!!!\n";
		}
		return data[i]; 
	    }

	    void update_idx(int idx, double val){ data[idx] = val; } // inlined function

	private:
	    int number_of_elements;
	    double * data;
    };

    Vector::~Vector (){ // Putting outside of class but inside namespace for example purposes
	std::cout << "Destroying vector of size " << number_of_elements << "\n";
	delete[] data;
    };

    // Constructor
    Vector::Vector(int num) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements << "\n";
	for (auto i = 0; i < num; ++i) {
	    data[i]=0; 
	}
    };

    // Overloaded Constructor
    Vector::Vector(int num, double init) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements 
	<<  " with elements initialised to " << init << "\n";
	for (auto i = 0; i < num; ++i) {
	    data[i]=init; 
	}
    };

} /* HPC */ 

// Outside namespace. For example purposes. Copy constructor
HPC::Vector::Vector(const Vector & old){
    std::cout << "Creating copy" << std::endl;
    number_of_elements = old.number_of_elements;
    data = new double [number_of_elements];
    for (auto i = 0; i < number_of_elements; i++) {
	data [i] = old.data[i];
    }
}

// Copy assignment operator
HPC::Vector& HPC::Vector::operator=(const Vector &invect){
    std::cout << "Copy assignment" << "\n";
    if(this == &invect){ 		// Are we copying same object to itself? If so, don't need to
        std::cout << "self copy!\n"; 	// do anything more. Could also actually be dangerous if we
	return *this;  			// are freeing memory in general case of a copy.
    } 					// Always CONSIDER self assignment check.
    				
    if(number_of_elements != invect.number_of_elements){
	delete[] data;
	data = new double[invect.number_of_elements];
	number_of_elements = invect.number_of_elements;
    }

    for (auto i = 0; i < number_of_elements; ++i) {
      data[i] = invect.data[i]; 
    }
    return *this;
}

int main()
{
    HPC::Vector A {3}; 
    HPC::Vector C {5, 10.0}; 	//Overloaded constructor

    for (auto i = 0; i < 5; ++i) {
	std::cout << "C[" << i << "]="<< C[i] << "\n";
    }

    C=A; 		// Copy assignment
    for (auto i = 0; i < 5; ++i) {
	std::cout << "C[" << i << "]="<< C[i] << "\n";
    }

    C=C;  		// Self assignment
    return 0;
}
