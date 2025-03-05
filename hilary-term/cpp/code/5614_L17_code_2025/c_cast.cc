#include <iostream>


int main()
{
    int a {3};
    int b {2};

    // double C {a/b};  // This actually warns
    double d = a/b;
    // prints d: 1
    std::cout << "d: " << d << '\n';

    double e = (double) a/b;
    // prints e: 1.5
    std::cout << "e: " << e << '\n';

    // No Error/Warning!!!!!
    char *f = (char *) &a;
    std::cout << "*f = " << *f << '\n';

    // This gives an error
//     char *g = static_cast<char *>(&a);

    return 0;
}
