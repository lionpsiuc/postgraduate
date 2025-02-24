#include <iostream>
#include <string>


template <typename T>
void generic_print(T& in){

    std::cout << "Generic print: Value is " << in << std::endl;
}

int main()
{

    int classnum = 5614;
    double dval  = 1.23;
    std::string mystr {"Hello World"};

    generic_print(classnum);
    generic_print(dval);

    return 0;
}
