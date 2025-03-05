#include <concepts>
#include <iostream>

auto gcd(auto a, auto b){
	if(b==0){
		return a;
	}
	return gcd(b, a%b);
}

int main()
{
	std::cout << gcd(70, 30) << '\n';
	return 0;
}
