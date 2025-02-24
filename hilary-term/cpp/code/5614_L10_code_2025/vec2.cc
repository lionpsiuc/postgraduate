#include <iostream>
#include <vector>


int main()
{

    int x {1};

    std::vector<int> A;
    A.resize(10); 
    std::cout << "1. A.size() = " << A.size() 
	<< "\nA.capacity() =" << A.capacity() << '\n';
    A.push_back(x);
    std::cout << "2. A.size() = " << A.size() 
	<< "\nA.capacity() =" << A.capacity() << '\n';
    A.shrink_to_fit();
    std::cout << "3. A.size() = " << A.size() 
	<< "\nA.capacity() =" << A.capacity() << '\n';
    A.emplace_back(x);
    std::cout << "4. A.size() = " << A.size() 
	<< "\nA.capacity() =" << A.capacity() << '\n';

    std::vector<std::vector<int>> I ;  // Create vector of vectors of ints
    for (int i = 0; i < 3; i++) {
	I.push_back(std::vector<int>(4)); // Add a vec of 4 ints each time.
    }

    // Create vec of 3 elements. Each element is a vec containing 4 ints.
    std::vector<std::vector<int>> B (3,std::vector<int>(4));  
    // Create vector of vector (2x3), initialised to 0
    std::vector<std::vector<int>> C (2,std::vector<int>(3,0));  

    C.resize(3);
    std::vector<int> tmp {1,2,3};
    C.push_back(tmp);

    return 0;
}
