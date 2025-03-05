#include <memory>
#include <iostream>


int main()
{
    int *a {new int {10}};

    std::unique_ptr<int> p1 {a};
    if(p1==nullptr){  	// Should not be null
	std::cout << __LINE__ << ": p1 = nullptr\n";
    }

    std::unique_ptr<int> p2  {std::move(p1)};
    if(p1==nullptr){  	// Should be null
	std::cout << __LINE__ << ": p1 = nullptr\n";
    }
    std::cout << "Value: p2 \t" << *p2 << '\n';

    // Will see the diff between move and reset later
    p2.reset(new int {2});
    std::cout << "Value: p2 \t" << *p2 << '\n';

    return 0;
}
