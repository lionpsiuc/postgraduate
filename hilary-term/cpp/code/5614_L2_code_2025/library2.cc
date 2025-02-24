// Library 2
#include <iostream>

void print_name(){
    std::cout << "This is from library 2" << std::endl;
    for (auto i=0; i <10; ++i) {
	std::cout << i << "\n";
    }
}

void library2_function() {
    // Do something useful
    //
    return;
}
