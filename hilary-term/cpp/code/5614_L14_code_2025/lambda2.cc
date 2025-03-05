#include <iostream>

auto func(int x){
    auto mylambda = [ x=2*x, y=x*x](int z){
       	std::cout << x << '\t' << y <<
	    '\t' << y*z << '\n';
    };
    return mylambda;
}

int main()
{
    auto L1 = func(4);
    auto L2 = func(10);

    L1(2);
    L1(5);
    L2(7);

    return 0;
}
