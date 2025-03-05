#include <iostream>

int main()
{
    int x {10};
    const int * y {&x};
    const int & rx {x};

    std::cout << "1) x: " <<  x << '\n';

    // *y = 20; 	// Error
    // rx = 30; 	// Error

    // int * z {y};     // Error
    int * z {const_cast<int *>(y)}; // OK
    *z = 20;

    std::cout << "2) x: " <<  x << '\n';
 
    const_cast<int &>(rx) = 30;
    std::cout << "3) x: " <<  x << '\n';

    return 0;
}
