#include <string>
#include <iostream>
#include <map>  	// Needed for std::map
#include <vector>  	// Needed for std::vector

class student_5614
{
public:
    student_5614 (std::string in)
       :	name {in}
    {
	std::cout << "Constructing base class " << name << '\n';
    }

    void add_marks (int assignment, double res){
	marks.insert(std::make_pair(assignment,res));
    }

    void print_marks() const{
	std::cout << name << ":" << '\n';
	for (auto i : marks)
	    std::cout << i.first << '\t' << i.second << '\n';
    }

private:
    const std::string name;
    std::map<int, double> marks;
};

class student_hpc : public student_5614
{
public:
    student_hpc (std::string name)
	: student_5614 {name}
    {
	std::cout << "Constructing student_hpc " << name << '\n';
    }

private:
    std::vector <int> other_courses;
};

class student_phd : public student_5614
{
public:
    student_phd (std::string name, std::string dept)
	: student_5614 {name}, department {dept}
    {
	std::cout << "Constructing student_phd " << name << '\n';
    }

private:
    std::string department;
};


int main()
{
    // Create object of base class
    student_5614 A {"Joe"};
    A.add_marks(0, 0.99);

    // Create object of derived class
    student_phd B {"John", "Physics"};
    B.add_marks(1,0.55);
    B.add_marks(2,0.67);

    // Create object of derived class
    student_hpc C {"Mary"}; 
    C.add_marks(1, 0.91);
    C.add_marks(2, 0.42);

    // print values for each
    A.print_marks();
    B.print_marks();
    C.print_marks();

    return 0;
}
