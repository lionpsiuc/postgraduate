#include <string>
#include <iostream>

class FuncObj
{
public:
    FuncObj (std::string x) : classname{x} {};
    void operator()(const std::string& Z, const int mark){
	std::cout << Z << " scored " << mark  <<
	    " for class: " << classname << '\n';
    }

private:
    std::string classname;
};

int main()
{
    // Create objects
    FuncObj cl1{"C++ programming"}; 		
    FuncObj cl2{"C programming"};

    // Call function object cl1
    cl1("John", 25);
    cl1("Mary", 50);
    cl1("Tom", 60);

    // Call function object cl2
    cl2("Mary", 75);
    
    return 0;
}
