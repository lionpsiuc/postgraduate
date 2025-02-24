/**
 * @file move.cc
 * @brief Simple example to show difference between deleting vs not defining move consructor
 *  	Relevant to L9 of C++.
 * @author R. Morrin
 * @version 1.0
 * @date 2024-02-23
 */
#include <iostream>


class Widget
{
    public:
	Widget () = default;
	Widget (const Widget&){ std::cout << "Cpy Ctor" << std::endl;}
	//Widget (Widget && in){ std::cout << "Move ctr " << std::endl;}
	Widget (Widget && in) = delete;

	~Widget () = default;

    private:
	/* data */
};

int main()
{
	Widget A;
	Widget B {std::move(A)};
	return 0;
}
