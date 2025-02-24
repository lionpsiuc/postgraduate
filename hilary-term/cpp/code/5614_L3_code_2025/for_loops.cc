#include <iostream> 			// Needed for std::cout
#include <vector> 			// Needed for std::vector

int main()
{
    // Standard for loop. Note generally use prefix increment in C++
    for (int i = 0; i < 5; ++i) {
	std::cout << "Hello World\n";
    }

    std::vector<int> v {1,2,3,4,5}; 	// We have not covered STL yet

    for (std::vector<int>::const_iterator it = v.cbegin(); it != v.cend(); ++it) {
	std::cout << *it << "\n";
    }

    std::cout << "\nUsing auto:" << std::endl;
    for (auto it2 = v.cbegin(); it2 != v.cend(); ++it2) { // Note auto
	std::cout << *it2 << "\n";
    }

    // Range-for was introduced in C++11. Like "foreach" statement.
    std::cout << "\nRange for:\n";
    for (int n : {5, 4, 3, 3, 1, 0}){ // the initializer may be a braced-init-list
	std::cout << n << "\n";
    }

    for (auto i : v) {
	std::cout << i << "\n";
    }
    return 0;
}
