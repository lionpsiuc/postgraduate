#include <valarray>
#include <iostream>

int main()
{
	std::valarray<int> A {1,2,3,4};
	std::valarray<int> B {5,6,7,8};
	std::valarray<int> C = 2*A+B;

	for (const auto& elem : C)
		std::cout << elem << '\n';

	return 0;
}

