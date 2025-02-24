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

    // This will override the base class function
    void print_marks() const override{
	std::cout << "Hides base class print_marks!\n";
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

private:
    std::string department;
};



//Exact same main function as in previous example
int main()
{
  // Create object of first derived class
  student_phd A {"John", "Physics"};
  A.add_marks(1,0.55);
  A.add_marks(2,0.67);

  // Create object of second derived class
  student_hpc B {"Mary"}; 
  B.add_marks(1, 0.91);
  B.add_marks(2, 0.42);


  std::cout << "\nSimply polymorphism example\n";
  // C is a pointer to base-class type
  student_5614 * C = &A;
  A.print_marks(); 	
  C->print_marks();

  C = &B;
  B.print_marks(); 	// Hides print_marks
  C->print_marks();

    return 0;
}
