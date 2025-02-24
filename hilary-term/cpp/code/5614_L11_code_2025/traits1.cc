/**
 * @file traits1.cc
 * @brief Example taken from https://en.cppreference.com/w/cpp/types/is_floating_point
 * @author R. Morrin
 * @version 1.0
 * @date 2022-03-02
 */


#include <iostream>
#include <type_traits>
 
class A {};
 
int main() 
{
    std::cout << std::boolalpha;
    std::cout << "      A: " << std::is_floating_point<A>::value << '\n';
    std::cout << "  float: " << std::is_floating_point<float>::value << '\n';
    std::cout << " float&: " << std::is_floating_point<float&>::value << '\n';
    std::cout << " double: " << std::is_floating_point<double>::value << '\n';
    std::cout << "double&: " << std::is_floating_point<double&>::value << '\n';
    std::cout << "    int: " << std::is_floating_point<int>::value << '\n';
    return 0;
}
