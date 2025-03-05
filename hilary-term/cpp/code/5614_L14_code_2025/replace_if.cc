#include <valarray>
#include <algorithm>
#include <iostream>
#include <functional>


int main()
{
    std::valarray<double> data {1.1, -0.6, -2.0, 3, 5};

    std::replace_if(std::begin(data), std::end(data),
	    std::bind(std::less<double>{}, std::placeholders::_1, 0), 0);

    for(const auto& elem : data){
	std::cout << elem << '\n';
    }

    // Example use of mathematical operations on valarrays



    return 0;
}
