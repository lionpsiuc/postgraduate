#include <string>
#include <iostream>

class student
{
    public:
	student (const std::string in, const double x)
	    : name{in}
	    , result{x}
	    {};

	void print_details() const { 
	    accessed++;  	// increment internal counter
	    std::cout << name << " scored " << result << "\n";
	}
	static void inc_tot() {++total_students;};
	static int get_tot() {return total_students;};

    private:
	const std::string name;
	const double result;
	mutable int accessed {0};
	// can do this since C++17. Note "inline" keyword.
	inline static int total_students {0};
};

int main()
{
    std::cout << "Total number of students is " <<  student::get_tot() << std::endl;

    student A { "Tom", 0.85};
    A.inc_tot(); 	// Increase class static member
    std::cout << "Total number of students is " <<  A.get_tot() << std::endl;
    
    student B {"Mary", 0.91};
    B.inc_tot(); 	// Increase class static member again
    std::cout << "Total number of students is " <<  B.get_tot() << std::endl;
    
    return 0;
}
