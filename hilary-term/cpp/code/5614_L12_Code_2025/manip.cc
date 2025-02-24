#include <iostream>
#include <iomanip>  // Needed for setprecision etc.

int main()
{

    double A {123.4567890};

    std::cout << std::setw(30) << std::setfill('-') <<'\n';
    std::cout << std::setw(15) << std::setfill(' ') << A << '\n';
    std::cout << std::fixed 		<< A << '\n';
    std::cout << std::scientific 	<< A << '\n';
    std::cout << std::setprecision(3) 	<< A << '\n';
    std::cout << std::setw(12) 		<< A << '\n';
    std::cout << std::setfill('*');
    std::cout << std::setw(12) 		<< A << '\n';
    std::cout << std::left;
    std::cout << std::setw(12)		<< A << '\n';
    std::cout << std::setw(30) << std::setfill('-') << '\n';
    
    return 0;
}
