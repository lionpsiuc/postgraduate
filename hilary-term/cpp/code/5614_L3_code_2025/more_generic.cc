/**
 * @file more_generic.cc
 * @brief  Quick example for L3 to show some generic programming concepts
 * 		You would not be expected to understand this.
 * 		Also, it is not necessarily the best way to write this code. It is an example
 * 		e.g. for printing, we could overload operator<< for the containers instead
 * 		and eliminating the loops could be argued to be unnecessary.
 *
 * 		Note. I don't have any error checking that matrix<->vector are appropriate sizes etc.
 *
 * @author R. Morrin
 * @version 1.0
 * @date 2023-01-31
 */
#include <iostream>
#include <array>
#include <algorithm>
#include <numeric>
#include <vector>

auto mat_vec(auto const & mat, auto const & vec){
    auto result {vec}; 					// Container to store result
    std::fill(std::begin(result), std::end(result), 0); 	// Fill results container with zeros
    auto res_it {std::begin(result)}; 			// Iterator to start of results container

    // This is a lambda expression. See later.
    auto ip = [&vec, &res_it](const auto & i){
	// std::inner_product. Compute dot product
	*res_it++ = std::inner_product(std::begin(i), std::end(i), std::begin(vec), 0.0);
    };

    // Do operation for each element in container
    std::for_each(std::begin(mat), std::end(mat), ip);

    return(result);
}

int main()
{
    // Call function for array^2 X array
    const std::array<const std::array<double,3>,3> A {{{1.0, 2.0, 3.0},
	{4.0, 5.0, 6.0},
	{7.0,8.0,9.0}}}; 	// array of arrays	
    const std::array<double,3> B {1.0, 2.0, 3.0}; 

    auto res {mat_vec(A,B)}; 	// Call function and store result
				// Print result
    std::for_each(std::cbegin(res), std::cend(res), [](auto const & x) { std::cout << x << '\t';}) ;
    //print newline
    std::cout << std::endl;

    // Call function for vector^2 X vector of ints
    const std::vector<std::vector<int>> A1 {{{1, 2}, {8,9}}};
    const std::vector<int> B1 {1, 3}; 
    auto res2 {mat_vec(A1,B1)};
    std::for_each(std::cbegin(res2), std::cend(res2), [](auto const & x) { std::cout << x << '\t';}) ;
    std::cout << std::endl;

    // Call function for vector^2 X array
    const std::array<double,2> B2 {4.0, 5.0};
    auto res3 {mat_vec(A1,B2)};
    std::for_each(std::cbegin(res3), std::cend(res3), [](auto const & x) { std::cout << x << '\t';}) ;
    std::cout << std::endl;

    return 0;
}
