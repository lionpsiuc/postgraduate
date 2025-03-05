#include <iostream>
#include <vector>

int main()
{
	std::vector<int> A {1, 2, 3, 4};

	// A.size() returns a size_t which is unsigned int
	for(int i = 0; i < (int) A.size(); ++i){
		std::cout << "A[" << i << "] = " << A.at(i) << '\n';
	}

	// A.size() returns a size_t which is unsigned int
	for(int i = 0; i <  static_cast<int>(A.size()); ++i){
		std::cout << "A[" << i << "] = " << A.at(i) << '\n';
	}

	// Better to do it this way
	for(auto i = 0U; i < A.size(); ++i){
		std::cout << "A[" << i << "] = " << A.at(i) << '\n';
	}

	// or even this way
	for(std::size_t i = 0; i < A.size(); ++i){
		std::cout << "A[" << i << "] = " << A.at(i) << '\n';
	}

	// or maybe this
	for(unsigned i = 0; i < A.size(); ++i){
		std::cout << "A[" << i << "] = " << A.at(i) << '\n';
	}

	// std::ssize() . Need C++20
	for(auto i = 0; i < std::ssize(A); ++i){
		std::cout << "A[" << i << "] = " << A.at(i) << '\n';
	}

	return 0;
}
