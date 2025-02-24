#include <iostream>
#include <complex>

template <typename T>
T mypr(T in){
    std::cout << "First\n";    
    return in;
}

template <typename T>
std::complex<T> mypr(std::complex<T> in) {
    std::cout << "Second\n";    
    return in;
}

double mypr(double in){
    std::cout << "Third\n";    
    return in;
}


int main ()
{
    std::complex<double> z {1.0, 2.0};   // defines 1.0 + 2.0i
    mypr(2); 		// Prints "First"
    mypr(2.0); 		// Prints "Third"
    mypr(z); 		// Prints "Second"
    return 0;
}
