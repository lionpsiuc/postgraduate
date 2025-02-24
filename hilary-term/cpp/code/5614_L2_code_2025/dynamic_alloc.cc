#include <iostream>
#include <memory> 		// Needed for unique_ptr
// Would need additional boost header files and libraries as well.

int main()
{
    // Allocate memory
    int *A = new int; 		// integer A is allocated on the heap
    int *B {new int[2]}; 	// memory is allocated for array of 2 ints
    int *C {new int[2]{8,9}}; 	// allocate and initialise (C++11)
    int *D {new int(5)}; 	// Allocates memory for integer and 
    				//   initialises it to value 5
    std::unique_ptr<int> E {new int{11}}; // Smart pointer! See later!!
    					  // (Even better way to do this.)
// boost::unique_ptr<int> F {new int{13} 	// An alternate implementation
						// pre-dating std::unique_ptr

    *A = 2;

    std::cout << "A: " << *A  << "\n"
	<< "B: " << B[0] <<", " << B[1] << "\n"
	<< "C: " << C[0] <<", " << C[1] << "\n"
	<< "D: " << *D << "\n"
	<< "E: " << *E << "\n";

    // Deallocate memory
    delete A;
    delete[] B; 	// B was allocated using new[]
    delete[] C; 	
    delete D;

    // Don't need to manually delete for smart pointers.
    return 0;
}
