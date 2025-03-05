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
	std::cout << "Constructing base class" << '\n';
    }

    void add_marks (int assignment, double res){
	marks.insert(std::make_pair(assignment,res));
    }

// Indicates that func. can be overridden in derived classes
    virtual void print_marks() const{
	std::cout << name << ":" << '\n';
	for (auto i : marks)
	    std::cout << i.first << '\t' << i.second << '\n';
    }

    virtual void print_details() = 0; // Pure virtual function.

private:
    std::string name;
    std::map<int, double> marks;
};

class student_hpc : public student_5614
{
public:
    student_hpc (std::string name)
	: student_5614 {name}
    {
	std::cout << "Constructing student_hpc" << '\n';
    }

    void print_details () final {
	    print_marks();
	std::cout << "Other Courses:\n";
	for (auto const& i: other_courses){
		std::cout << i << std::endl;
	}
    }

    void add_other_course(int const i){
	other_courses.push_back(i);
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
	std::cout << "Constructing student_phd" << '\n';
    }

    void print_details(){
	    print_marks();
	std::cout << "Department:\t" << department << '\n';

    }

private:
    std::string department;
};

void dump_details(student_5614 *in[], int const N){
	for (int i = 0; i < N; ++i) {
		in[i]->print_details();	
	}
}	


//Exact same main function as in previous example
int main()
{
  // student_5614 {"Joe"};  	// Not allowed. Abstract Base class
  // Create object of first derived class
  student_phd A {"John", "Physics"};
  A.add_marks(1,0.55);
  A.add_marks(2,0.67);

  // Create object of second derived class
  student_hpc B {"Mary"}; 
  B.add_marks(1, 0.91);
  B.add_marks(2, 0.42);
  B.add_other_course(5613);
  B.add_other_course(5612);


  std::cout << "\nSimply polymorphism example\n";
  student_5614 *C[2];

  C[0] = &A;
  C[1] = &B;

  dump_details(C,2);

    return 0;
}
