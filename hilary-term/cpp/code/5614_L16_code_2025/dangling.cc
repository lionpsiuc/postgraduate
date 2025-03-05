#include <iostream>

int main()
{
    int *b {nullptr};

    { // Create inner scope for a
	int *a {new int {10}};
	b = a;

	std::cout << "a is stored at address " << &a <<
	    " and holds value " << a <<
	    ". The value pointed to by a is " << *a << '\n';
	std::cout << "b is stored at address " << &b <<
	    " and holds value " << b <<
	    ". The value pointed to by b is " << *b << '\n';

	delete a;
    }

    int * c {new int {20}};
    std::cout << "c is stored at address " << &c <<
	" and holds value " << c << '\n';

    // We don't have a way of checking whether the object b 
    // pointed to still exists
    std::cout << "b is stored at address " << &b <<
	" and holds value " << b <<
	". The value pointed to by b is " << *b << '\n';


    return 0;
}
