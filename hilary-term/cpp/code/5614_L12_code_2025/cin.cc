#include <iostream>


int main()
{
    std::string name;
    int age;

    std::cout << "Please enter a name and age" << std::endl;
    std::cin >> name >> age;;

    std::cout << name <<" is " << age << " years old\n";
    return 0;
}
