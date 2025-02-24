/**
 * @file vector_move.cc
 * @brief Code for 5614 L5 example
 * 	Included in L8 to show copy elision
 * @author R. Morrin
 * @version 2.1
 * @date 2021-02-24
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
	    Vector incremented_copy(const Vector in);
	    Vector(Vector&& a); 		// Move constructor

	    double& operator[](int i) { 	//operator[] must be inlined
		if(i >= number_of_elements){
		    std::cerr << "ERROR: Accesing beyond bounds!!!!\n";
		}
		return data[i]; 
	    }

	    void update_idx(int idx, double val){ data[idx] = val; } // inlined function
	    int size(){ return(number_of_elements); }

	private:
	    int number_of_elements;
	    double * data;
    };

    // Destructor
    Vector::~Vector (){ // Putting outside of class but inside namespace for example purposes
	std::cout << "Destroying vector of size " << number_of_elements << "\n";
	delete[] data;
    };

    // Constructor taking one integer parameter for Vector class
    Vector::Vector(int num) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements << "\n";
	for (auto i = 0; i < num; ++i) {
	    data[i]=0; 
	}
    };

    // Overloaded constructor taking two parameters for Vector class
    Vector::Vector(int num, double init) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements 
	    <<  " with elements initialised to " << init << "\n";
	for (auto i = 0; i < num; ++i) {
	    data[i]=init; 
	}
    };

} /* HPC */ 

// Outside namespace. For example purposes.
HPC::Vector::Vector(const Vector & old){
    std::cout << "Copy Constructor" << std::endl;
    number_of_elements = old.number_of_elements;
    data = new double [number_of_elements];
    for (auto i = 0; i < number_of_elements; i++) {
	data [i] = old.data[i];
    }
}

// Copy assignment operator
HPC::Vector& HPC::Vector::operator=(const Vector &invect){
    std::cout << "Copy assignment" << "\n";
    if(this == &invect){ 		// Are we copying same object to itself?
        std::cout << "self copy!\n"; 	// If so, don't need to do anything more. Could also
	return *this;  			// actually be dangerous if we are freeing memory in general case.
    }
    				
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

// Function used to illustrate need for a move.
HPC::Vector incremented_copy(const HPC::Vector in){
    std::cout << "Entering incremented copy" << std::endl;
    HPC::Vector X {in};

    for (auto i = 0; i < X.size(); ++i) {
	X.update_idx(i, X[i]+1); 
    }
    std::cout << "Leaving incremented copy" << std::endl;
    //return std::move(X);    // Uncomment to turn off RVO
    return X;
}

// Move constructor Function definition - "grab the elements" from "a"
HPC::Vector::Vector(HPC::Vector&& a)
       	: number_of_elements{a.number_of_elements}
	, data{a.data} {
    std::cout << "Move constructor"  << std::endl;
    a.data = nullptr; // now a has no elements
    a.number_of_elements = 0;
}

int main()
{
    HPC::Vector C {5, 10.0}; 	//Overloaded constructor

    for (auto i = 0; i < C.size(); ++i) {
	std::cout << "C[" << i << "]="<< C[i] << std::endl;
    }

    HPC::Vector D {incremented_copy(C)};
    for (auto i = 0; i < D.size(); ++i) {
	std::cout << "D[" << i << "]="<< D[i] << std::endl;
    }

    /*
     *HPC::Vector E {5}; 	
     *HPC::Vector F {std::move(E)}; // std::move just returns an rvalue ref
     */
    // or HPC::Vector F = std::move(E);
    return 0;
}
