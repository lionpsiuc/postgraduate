/**
 * @file rangess2.cc
 * @brief Simple example of using ranges. Will need C++20
 * 		Modification of ranges.cc to use ranges::for_each to print
 * @author R. Morrin
 * @version 1.0
 * @date 2022-03-18
 */

#include <iostream>
#include <algorithm>
#include <vector>
#include <ranges>

bool positive(int in){
	return in>0;
}

bool even(int in){
	return in%2==0;
}

int square(const int x){
	return x*x;
}

void print(const int& y){
	std::cout << y << '\n';
}

int main()
{
	std::vector<int> v1 {1, -2, 3, 4};

	auto filtered = v1 | std::ranges::views::filter(positive);
	std::ranges::for_each(filtered, print);
	std::cout << "\nFiltered and transformed:\n";
	auto filt_and_trans = v1 | std::views::filter(even) | std::views::transform(square);
	std::ranges::for_each(filt_and_trans, print);

	// Without ranges
	std::for_each(std::begin(filt_and_trans), std::end(filt_and_trans), print);
	return 0;
}
