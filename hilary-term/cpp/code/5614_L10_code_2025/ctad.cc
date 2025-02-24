#include <iostream>
#include <vector>


int main()
{
	std::string s {"HPC C++ Programming"};
	std::pair<int, std::string> p1 {1, s };
	std::pair p2 {2, s};
	
	std::vector<int> v1 {1, 2, 3};
	std::vector v2 {4, 5, 6};

	std::vector<double> v3 {4.0, 5, 6};
	// std::vector<int> v4 {4.0, 5, 6}; // Error. Narrowing.
	// std::vector v5 {4.0, 5, 6}; 	// Error
	
	return 0;
}
