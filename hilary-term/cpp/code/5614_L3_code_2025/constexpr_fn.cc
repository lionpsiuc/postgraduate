#include <iostream>

// C++11 introduced constexpr
constexpr double sq(const double x){
	return(x*x);
}

constexpr int fact(const int n){
    // Conditional operator 
    return (n>1) ? n * fact(n-1) : 1;
}

int main()
{
    const double a {1.2};
    const int b {3};

    std::cout << sq(a)  << std::endl; 
    std::cout << fact(b)  << std::endl; 
    return 0;
}
