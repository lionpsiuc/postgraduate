#include <iostream>

// General template
template<typename>
struct is_const{
    static const bool value = false;
};    // Version 1

// Specialised for const
template<typename Tp>
struct is_const<Tp const>{ 
    static const bool value = true;
};    // Version 2

// Helper variable template
template<typename T >
inline constexpr bool is_const_v = is_const<T>::value;

int main()
{
    std::cout << std::is_const_v<const double> << std::endl; // 1

    int a;
    const int b = 3;
    std::cout << is_const<int>::value << '\n';    // 0
    std::cout << is_const<decltype(a)>::value << '\n';    // 0
    std::cout << is_const_v<decltype(b)> << '\n';    // 1
    return 0;
}
