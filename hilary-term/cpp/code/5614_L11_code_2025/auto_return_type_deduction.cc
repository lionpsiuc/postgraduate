#include <iostream>

// Can have declarations, but still need 
//    definition before function is used
// auto fn11(std::string)->std::string;
// auto fn14(std::string);

// C++11 way
auto fn11(std::string in)->std::string {
    std::string copied{in};
    std::cout << "Inside f11\n";
    return in + " returned from f11";
}

// C++14 way
auto fn14(std::string in) {
    std::string copied{in};
    std::cout << "Inside f14\n";
    return in + " returned from f14";
}

int main()
{
    std::string name {"5614 C++"};

    std::cout << fn11(name) << '\n';
    std::cout << fn14(name) << '\n';

    return 0;
}


