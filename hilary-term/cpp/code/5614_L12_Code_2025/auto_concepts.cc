#include <concepts>
#include <iostream>

template <typename T>
concept Integral=std::is_integral<T>::value;

auto gcd(Integral auto a, Integral auto b){ // Constrained
// auto gcd(auto a, auto b){
	if(b==0){
		return a;
	}
	return gcd(b, a%b);
}

int main()
{

	std::cout << gcd(70, 30) << '\n';
	//std::cout << gcd(70.1, 30) << '\n';
	return 0;
}
