#include <string>
#include <iostream>
#include <functional> 	// Needed for bind

void bfunc1(std::string name){
  std::cout << "The person's name is "
    << name <<'\n';
}

int main()
{

  std::string person {"John"};
  auto f1 = std::bind(bfunc1, person);

  bfunc1(person);
  //Same result with bound function
  f1();

  return 0;
}
