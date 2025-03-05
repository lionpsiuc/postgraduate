/**
 * @file lambda_capture.cc
 * @brief Simple example to show difference between capture by value and capture by reference
 * 	MAP55614 lecture on lambda expressions
 * @author R. Morrin
 * @version 1.0
 * @date 2022-03-18
 */

#include <iostream>


int main(void)
{
    int a {10};

    auto l1 = [a](int x){  std::cout << x*a << std::endl;};
    auto l2 = [&a](int x){ std::cout << x*a << std::endl;};

    a = 20;

    l1(10);
    l2(10);

    return 0;
}
