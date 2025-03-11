/**
 * @file widget2.cc
 * @brief Used for written question in 5614 C++ Assignment 3, 2025
 * @author R. Morrin
 * @version 1.0
 * @date 2025-02-22
 */
#include <iostream>
#include <list>
#include <numeric>


/**
 * @brief Structure to hold a Widget.
 */
struct Widget
{
	Widget (std::size_t in) 
		: vals (in) 
		  , original_N {in} {
			  // Populate with 0,..,in-1
			  std::iota(vals.begin(), vals.end() ,0);
		  }

	// ~Widget(){ std::cout << "Deleting Widget\n";}

	std::list<int> vals;

	std::size_t original_N;
	// std::size_t const original_N;
};


int main()
{
	Widget X {10};
	Widget Y {X};
	Widget Z {std::move(X)};
	Z=Y;
	X=std::move(Z);

	for (auto const & i : X.vals) {
		std::cout << i << '\n';
	}

	return 0;
}
