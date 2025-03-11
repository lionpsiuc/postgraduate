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
 * @brief Structure to hold a Widget2.
 */
struct Widget2
{
	Widget2 (std::size_t in) 
		: vals (in) 
		  , original_N {in} {
			  std::cout << "Constructor" << std::endl;
			  // Populate with 0,..,in-1
			  std::iota(vals.begin(), vals.end() ,0);
		  }

	Widget2(Widget2 const& rhs){  	// Copy Constructor
		original_N = rhs.original_N;
		vals = vals;
		std::cout << "Copy Constructor" << std::endl;
	}

	Widget2(Widget2 && rhs){ 	// Move Constructor
		original_N = rhs.original_N;
		vals = std::move(rhs.vals);
		std::cout << "Move Constructor" << std::endl;
	}

	std::list<int> vals;
	private:
	std::size_t original_N;
};

/**
 * @brief Function which makes a copy of a Widget2, increments all the values, and returns the copy
 *
 * @param[in] w reference to the source Widget2
 *
 * @return The increment copy of the original Widget2
 */
Widget2 incremented_copy(Widget2 const& w){
	std::cout << "Entering incremented Copy" << std::endl;
	Widget2 incremented {w};
	for (auto &i : incremented.vals) {
		i++;
	}
	std::cout << "Leaving incremented Copy" << std::endl;
	return(incremented);
	//  return(std::move(incremented));
}

int main()
{
	Widget2 X {10};
	Widget2 W {incremented_copy(X)};

	for (auto const & i : W.vals) {
		std::cout << i << '\n';
	}

	return 0;
}
