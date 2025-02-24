#include <iostream>

int A = 1; 		// Global variables
int B = 1;

int main()
{
    int A = 10; 	// Local A hides global A
    A++; 		
    std::cout << "  A = " << A << std::endl;

    int B = 20; 	// Local B hides global B
    ::B++;
    std::cout << "::B = " << ::B << std::endl;
    std::cout << "  B = " << B << std::endl;

    return 0;
}
