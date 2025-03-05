#include <iostream>
#include <valarray>

int main()
{
    std::valarray<double> A {1.1, -0.6, -2.0, 3, 5};
    for (const auto elem : A){
	std::cout << elem << '\n';
    }
    std::cout << '\n';

    A[A<=0] = 0; // Perform masking

    for (auto i = std::begin(A); i < std::end(A); ++i){
	std::cout << *i << '\n';
    }

    return 0;
}
