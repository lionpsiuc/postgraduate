#include <iostream>
#include <array>
#include <algorithm>
#include <functional>

int main()
{
    std::array<int, 5> arr {4, 2, 1, 0, 3}; 

    std::cout << "Original array\n";
    //for (auto i: arr) {
    for (std::array<int,5>::iterator it = arr.begin(); it!= arr.end(); ++it){
	std::cout << *it << ' ';
    }

    // define a lambda expression. Easier to use auto here!
    // Second option used std::function
    // Third is C way. Works for lambda with no captures
    // auto comp = [](int a, int b){return a<b;};
    // std::function<bool(int, int)> comp = [](int a, int b){return a<b;};
    bool (*comp)(int, int) = [](int a, int b){return a<b;};
    // Sort using lambda to compare
    std::sort(std::begin(arr), std::end(arr), comp);
    // Could also just use it directly
    // std::sort(std::begin(arr), std::end(arr), [](int a, int b){return a<b;});
    
    std::cout << "\n\nAfter sorting\n";
    for (const auto& i: arr) {
	std::cout << i << ' ';
    }
    std::cout << '\n';

    return 0;
}
