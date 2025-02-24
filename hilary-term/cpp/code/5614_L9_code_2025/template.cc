#include <iostream>

/*
 double square (const double in){
    double tmp = in*in;
    std::cout << in << " squared equals " << tmp << '\n';
    return tmp;
}


int square (const int in){
    int tmp = in*in;
    std::cout << in << " squared equals " << tmp << '\n';
    return tmp;
}
*/

template <typename T>
T square (T in) {
    T tmp = in*in;
    std::cout << in << " squared equals " << tmp << '\n';
    return tmp;

}

int main()
{
	square (10.0);    
	square (2);
    return 0;
}
