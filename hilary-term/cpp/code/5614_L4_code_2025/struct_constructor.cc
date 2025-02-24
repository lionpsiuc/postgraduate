#include <iostream>
struct S {
    int n;
    // constructor declaration
    S(int); 
    // constructor definition.
    S() : n(7) { 
	std::cout << "Calling default constructor"  << "\n";
    } 
    // ": n(7)" is the initializer list
    // ": n(7) {}" is the function body
};
//
// constructor definition. ": n{x}" is the initializer list
S::S(int x) : n{x} { 
    std::cout << "Calling contructor for n = " << n << "\n";
}

int main()
{
    std::cout << "Creating object s" << std::endl;
    S s; 	// calls S::S()

    std::cout << "Creating object s2" << std::endl;
    S s2(10); 	// calls S::S(int)

    S s3 {20}; 	//  Uniform initialisation
    return(0);
}
