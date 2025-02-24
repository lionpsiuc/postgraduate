
#include <iostream>

int main()
{
    int V[] {2,1,4,6,8};

    for (auto i : V) {
	i+=10;
	std::cout << i << "\t";
    }
    std::cout << std::endl;

    for (auto i : V) {
	std::cout << i << "\t";
    }
    std::cout << std::endl;

    for (auto &i : V){
	i+=10;    
    }

    for (auto i : V)  
	std::cout << i << "\t";
    std::cout << std::endl;


    // Using a reference
    for (auto &&i : V) {  	// This is actually a forwarding reference not an rvalue reference
	i+=10;
    }

    for (const auto &i : V) {
	std::cout << i << "\t";
    }
    std::cout << std::endl;



    return 0;
}
