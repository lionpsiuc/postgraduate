#include <iostream>

template <typename T>
T add(T x, T y)
{
	std::cout << "Normal template\n";
	return x+y;
}

// specialized function
template<>
char add<char>(char x, char y)
{
	std::cout << "Specialized template\n";
	int i = x-'0';
	int j = y-'0';
	return i+j;
}

int main()
{
	//Normal templated version for int
	int a = 1, b = 2;
	std::cout << add(a, b) << '\n';

	//Normal templated version for double
	double c = 3.0, d = 5.5;
	std::cout << add(c,d) << '\n';

	//Specialised template version
	char e='e', f='f';
	std::cout << add(e,f);

	return 0;
}
