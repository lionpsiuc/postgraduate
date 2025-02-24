#include <string>
#include <iostream>

class student
{
    public:
	student (const std::string in, const double grade)
	    : name{in}
	, result{grade}
	{};

	void print_details() const { 
	    accessed++;  	// increment internal counter
	    std::cout << name << " scored " << result
		<< " \t(Internal counter " << accessed<< ")\n";
	}

    private:
	const std::string name;
	const double result;
	mutable int accessed = 0;
};


int main()
{
    student A { "Tom", 0.85};
    A.print_details(); // internal counter will hold 1 
    A.print_details(); // internal counter will hold 2

    const student B{"Mary", 0.99};
    B.print_details(); // internal counter will hold 1 
    B.print_details(); // internal counter will hold 2

    return 0;
}
