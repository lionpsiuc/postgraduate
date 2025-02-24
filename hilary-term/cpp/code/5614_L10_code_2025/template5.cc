#include <iostream>
#include <string>

class student
{
    public:
	student (std::string n, double x)
	    : name{n}
	, grade{x}
	{};
	// Declared as friend so that it can access private members for printing.
	friend std::ostream& operator<<(std::ostream& os, const student& x);

    private:
	std::string name;
	double grade;
};

// Overload stream operator << to print an object of type student
std::ostream& operator<<(std::ostream& os, const student& x)
{
    os << "\nName is " << x.name << '\n';
    os << "Grade is " << x.grade << '\n';
    return os;
}

template <typename T>
void generic_print(T& in){

    std::cout << "Generic print: Value is " << in << std::endl;
}

int main()
{

    /*
    int classnum = 5614;
    double dval  = 1.23;
    std::string mystr {"Hello World"};

    generic_print(classnum);
    generic_print(dval);
    generic_print(mystr);
    */

    student st1 {"John", 0.90};
    generic_print(st1);

    return 0;
}
