#include <iostream>
struct S {
    double n;  // NOTE this time a double
    //
    // default constructor definition.
    S() : n(7.0) { 
	std::cout << "Calling default constructor. n = "  << n << "\n";
    } 
    // Writing constructor
    S(double);
    
    // disabling constructor for int
    S(int) = delete;

    // destructor declaration
    ~S(){
	std::cout << "Calling destructor n="  << n << "\n";
    }
};
//
// constructor definition. ": n{x}" is the initializer list
S::S(double x) : n{x} { 
    std::cout << "Calling contructor for n = " << n << "\n";
}

int main()
{
    S s; 	// calls S::S(). Default constructor

    {
	std::cout << "Creating object s2" << "\n";
    //    S s2(11); 	// Now will not work for int!
	S s3(10.0); 	// This is fine
    } // destructor will get called here.

    std::cout << "\ns4\n";
    S s4 {20.0}; 	
    return(0);
}
