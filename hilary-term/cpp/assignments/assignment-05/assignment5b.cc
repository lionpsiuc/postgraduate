/**
 * @file assignment5b.cc
 * @brief Assignment 5 for 5614. Writing your own shared_ptr class
 * @author R. Morrin
 * @version 1.0
 * @date 2025-03-03
 */

#include <iostream>
#include <iomanip>
#include "shared_ptr.h"

int main()
{
	std::size_t sw {60}; // Width of lines for formatting.

	// Construct shared_ptr to double
	std::cout << std::setfill('-') << std::setw(sw) << '\n' << std::setfill(' ');
	std::cout << "Constructing A\n";
	HPC::shared_ptr<double> A {new double {10}};
	std::cout << "A.get() = " << A.get() << '\n'
		<< "A.use_count() = " << A.use_count() << '\n'
		<< "*A = " << *A << '\n';

	// Copy Construct
	std::cout << '\n' << std::setfill('-') << std::setw(sw) << '\n' << std::setfill(' ');
	std::cout << "Copy Constructing B from A\n";
	HPC::shared_ptr B {A}; 			// Uses CTAD
	std::cout << "B.get() = " << B.get() << '\n'
		<< "B.use_count() = " << B.use_count()  << '\n'
		<< "*B = " << *B << '\n';

	// Change via B and print via A
	*B = 100;
	std::cout << "*A = " << *A <<  '\n';


	// Copy assign
	std::cout << '\n' << std::setfill('-') << std::setw(sw) << '\n' << std::setfill(' ');
	std::cout << "Constructing C\n";
	HPC::shared_ptr C {new double {20}};

	std::cout << "Before Assignment:"
		<< "\nC.get() = " << C.get()
		<< "\nC.use_count() = " << C.use_count()
		<< "\n*C = " << *C << '\n';

	std::cout << "Copy Assignment C from A\n";
	C = A;

	std::cout << "After Assignment:" 
		<< "\nC.get() = " << C.get()
		<< "\nC.use_count() = " << C.use_count()
		<< "\n*C = " << *C << '\n';

	std::cout << '\n' << std::setfill('-') << std::setw(sw) << '\n' << std::setfill(' ');
	std::cout << "End of main()\n";

	return 0;
}
