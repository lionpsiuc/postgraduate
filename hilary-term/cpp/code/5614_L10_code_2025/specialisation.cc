#include <iostream>

template<int N>
constexpr int fac()
{
    return N*fac<N-1>();
}

template<>
constexpr int fac<1>()
{
    return 1;
}

int main()
{
    constexpr int x5 = fac<5>();

    std::cout << "5! = " << x5 << '\n';
    return 0;
}
