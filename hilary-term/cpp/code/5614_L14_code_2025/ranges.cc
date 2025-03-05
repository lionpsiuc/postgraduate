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

int main()
{
    std::vector<int> v1 {1, -2, 3, 4};

    auto filtered = v1 | std::ranges::views::filter(positive);

    for(const auto i: filtered){
	std::cout << i << '\n';
    }

    std::cout << "\nFiltered and transformed:\n";
    auto filt_and_trans = v1 | std::views::filter(even) | std::views::transform(square);
    for(const auto i: filt_and_trans){
	std::cout << i << '\n';
    }
    return 0;
}
