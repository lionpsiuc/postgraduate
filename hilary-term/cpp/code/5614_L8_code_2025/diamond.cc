#include <string>
#include <iostream>
class Person
{
  public:
    Person () { std::cout << "Default for Person\n";};
    Person(int n, std::string X)
      : age {n}
    , name {X}
    {
      std::cout << "Overloaded for Person\n";
    }

  private:
    int age;
    std::string name;
};


class Student : public Person
{
  public:
    Student (int m, std::string Y)
      : Person {m, Y}
    {
      std::cout << "Constructing for Student\n";
    }
};

class Teacher : public Person
{
  public:
    Teacher (int m, std::string Y)
      : Person {m, Y}
    {
      std::cout << "Constructing for Teacher\n";
    }
};

class Tutor : public Student, public Teacher
{
  public:
    Tutor (int x, std::string Y)
      : Student {x, Y}
    , Teacher {x, Y}
    {
      std::cout << "Constructing Tutor\n";
    }

};


int main()
{
  Tutor {30, "Tom"};
  return 0;
}
