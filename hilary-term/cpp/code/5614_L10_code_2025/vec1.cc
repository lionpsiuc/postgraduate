#include <iostream>
#include <vector>


int main()
{
	std::vector<int> A (2);  // Create vector of 2 integers
	std::vector<int> B {2};  // Create vector with one integer=2
	std::vector<int> C (3,4);  // Create vector with three integers initialised to value 4
	std::vector<int> D {3,4};  // Create vector with two integer elements - 3,4
	std::vector 	 E (5,6);  // Same as above but taking advantage of CTAD
	std::vector 	 F {5,6}; 

	// Declare and initialise vector-of-vector-of-integers. like a 2D structure
	std::vector<std::vector<int>> H  = { {0, 1},
					     {2, 3, 4},
					     {5}};

	H.clear();
	return 0;
}
