#include <iostream>
#include <string>
#include <memory>

struct Person
{
    Person (std::string n) : name{n}{
    std::cout << "Constructing " << name << '\n';
    };
    ~Person (){
	std::cout << "Destructor for " << name << '\n';
    };

    std::string name;
    std::shared_ptr<Person> neighbour;
};


int main()
{
   auto p1 = std::make_shared<Person>("John"); 
   auto p2 = std::make_shared<Person>("Mary"); 
   auto p3 = std::make_shared<Person>("Pat"); 

   p1->neighbour = p2;
   p2->neighbour = p3;
   //p3->neighbour = p1;

   std::cout << "John use count = " << p1.use_count() << '\n'
    << "Mary use count = " << p2.use_count() << '\n'
    << "Pat use count = " << p3.use_count() << '\n';

    return 0;
}
