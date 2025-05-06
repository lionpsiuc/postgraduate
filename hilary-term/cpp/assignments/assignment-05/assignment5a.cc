/**
 * @file assignment4a.cc
 * @brief  Main function for assignment4a. Assignment 5, 5614
 * 	This should work with the HPC::unique_ptr class you write
 * 	
 * @author R. Morrin
 * @version 2.0
 * @date 2025-03-03
 */
#include "unique_ptr.h"
#include <memory>

int main(void)
{
	// Constructor
	HPC::unique_ptr<double> A {new double(1.0)};
	HPC::unique_ptr A2 {new double(2.0)}; 		// Uses CTAD

	// error: no matching function for call to ‘HPC::unique_ptr::unique_ptr()’
	 /*
	  *HPC::unique_ptr<double> B ;   	
	  */

	// error: use of deleted function ‘HPC::unique_ptr::unique_ptr(const HPC::unique_ptr&)’
	/*
	 * HPC::unique_ptr C {A};   	
	 */

	HPC::unique_ptr D {new double(3.0)};

	// error: use of deleted function ‘HPC::unique_ptr& HPC::unique_ptr::operator=(const HPC::unique_ptr&)’
	/*
	 * D = A; 			
	 */

	// Move Constructor
	HPC::unique_ptr E {std::move(D)};

	// Overloaded dereference operator
	std::cout << "E = " << *E << std::endl;
	*E=4;
	std::cout << "E = " << *E << std::endl;

	// Move assignment
	D=std::move(E);
	std::cout << "D = " << *D << std::endl;

	return 0;
}
