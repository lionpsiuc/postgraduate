#include <iostream>

// C++20 introduced consteval
consteval double sq(const double x){
	return(x*x);
}

consteval int fact(const int n){
    // Conditional operator 
    return (n>1) ? n * fact(n-1) : 1;
}

int main()
{
    constexpr double a {1.2};
    // Note const-qualified integral type
    const int b {3}; 

    std::cout << sq(a)  << std::endl; 
    std::cout << fact(b)  << std::endl; 
    return 0;
}
