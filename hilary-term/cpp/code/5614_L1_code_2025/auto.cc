/**
 * @file auto.cc
 * @brief Note that this program needs C++11 or later
 * Short example code to show simple uses of auto
 * MAP55614 L1. 
 * @author R. Morrin
 * @version 3.0
 * @date 2025-01-31
 */



#include <iostream> 			// Needed for cout
#include <typeinfo> 			// Needed for typeid()

int main()
{

    auto v1 = 1; 			// v1 is an integer
    auto v2 = 2.0; 			// v2 is a double
    auto v3 = 3.0f; 			// v3 is a float
    auto v4 = 1L; 			// v4 is a long
    auto v5 = 1.0L; 			// v5 is a long double

    std::cout << "v1 is type " << typeid(v1).name() << "\n"
	<< "v2 is type " << typeid(v2).name() << "\n"
	<< "v3 is type " << typeid(v3).name() << "\n"
	<< "v4 is type " << typeid(v4).name() << "\n"
	<< "v5 is type " << typeid(v5).name() << "\n" << std::endl;


    for (auto i = 0; i < 5; ++i) {  	// auto commonly used in loops
       std::cout << "Iteration " << i << std::endl; 	
    }
    return 0;
}

