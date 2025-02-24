/**
 * @file sfinae.cc
 * @brief Explanation of SFINAE for L10 5614
 * @author R. Morrin
 * @version 1.0
 * @date 2022-03-03
 */
#include <iostream>

template <typename T>
struct super_int {
    int val;
    using type = T;
};


template <typename T>
typename T::type myprint(T in) {
    std::cout << "templated\n";
    return in.val;
};

int myprint (int in){
    std::cout << "regular\n";
    return in;
};


int main()
{
    super_int<int> A {};
    int B{};

    myprint(A);
    myprint(B);

    return 0;
}
