/**
 * @file vector_swap.cc
 * @brief  Some code for 5614 L6 2021
 * 	Copy and move semantics.
 * @author R. Morrin
 * @version 2.0
 * @date 2021-02-16
 */
#include <iostream>

namespace HPC
{
    class Vector
    {
	public:
	    Vector ()  				// Default constructor
		: number_of_elements {0}
		, data {nullptr}
	    { std::cout << "Default Constructor\n"; }; 
	    Vector (int num); 			// Constructor declaration
	    Vector (int num, double init); 	// Overloaded onstructor declaration
	    Vector(const Vector & old); 	// Copy constructor declaration
	    ~Vector (); 			// Destructor declaration
	    Vector& operator=(const Vector &in);// Copy assignment operator 
	    Vector incremented_copy(const Vector in); // You can ignore this for L6. I might come back to it later
	    Vector(Vector&& a); 		// Move constructor
	    Vector& operator=(Vector&& a); 	// Move assignment

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

    Vector::~Vector (){ // Putting outside of class but inside namespace for example purposes
	std::cout << "Destroying vector of size " << number_of_elements << "\n";
	delete[] data;
    };

    Vector::Vector(int num) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements << "\n";
	for (auto i = 0; i < num; ++i) {
	    data[i]=0; 
	}
    };

    Vector::Vector(int num, double init) : number_of_elements {num}, data {new double [num]}{
	std::cout << "Constructing vector of size " << number_of_elements 
	    <<  " with elements initialised to " << init << "\n";
	for (auto i = 0; i < num; ++i) {
	    data[i]=init; 
	}
    };

    void swap(Vector &A, Vector &B){
	std::cout << "Swapping\n";
	Vector temp {A};
	A = B;
	B = temp;
    }
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

// Simple function to increment values of a Vector
// Actually removed from slide so you can ignore!
// I may come back to it later for a different example.
HPC::Vector incremented_copy(const HPC::Vector in){
    HPC::Vector X {in};

    for (auto i = 0; i < X.size(); ++i) {
	X.update_idx(i, X[i]+1); 
    }
    return X;
}

// Move constuctor
HPC::Vector::Vector(HPC::Vector&& a) :number_of_elements{a.number_of_elements}, data{a.data} {// "grab the elements" from a
    std::cout << "Move constructor"  << std::endl;
    a.data = nullptr; // now a has no elements
    a.number_of_elements = 0;
}

// Move assignment operator
HPC::Vector& HPC::Vector::operator=(Vector && rhs){
    std::cout << "Move assignment operator" << std::endl;
    if( this != &rhs ){
	delete[] data;
	data = rhs.data;
	number_of_elements = rhs.number_of_elements;
	rhs.data = nullptr;
	rhs.number_of_elements = 0;
    }
    return *this;
}

int main()
{
    HPC::Vector A {5, 10.0}; 	
    HPC::Vector B {2, 5};

    for (auto i = 0; i < A.size(); ++i)
       std::cout << "\tA[" << i <<"] = " << A[i]; 
    std::cout << "\n";

    for (auto i = 0; i < B.size(); ++i)
       std::cout << "\tB[" << i <<"] = " << B[i]; 
    std::cout << "\n";

    HPC::swap(A,B); 			// Call swap 

    for (auto i = 0; i < A.size(); ++i)
       std::cout << "\tA[" << i <<"] = " << A[i]; 
    std::cout << "\n";

    for (auto i = 0; i < B.size(); ++i)
       std::cout << "\tB[" << i <<"] = " << B[i]; 
    std::cout << "\n";

    return 0;
}
