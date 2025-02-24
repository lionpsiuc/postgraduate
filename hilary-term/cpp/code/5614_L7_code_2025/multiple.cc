#include <string>
#include <iostream>

class base_A
{
  public:
    base_A (int in) 
      : number_of_A {in}
    {
      std::cout << "Calling base_A constructor for " << in << '\n';
    }

    ~base_A (){
      std::cout << "Destroying base_A " << number_of_A << '\n';
    }

  private:
    int number_of_A;
};

class base_B
{
  public:
    base_B (std::string in)
      : name {in}
    {
      std::cout << "Calling base_B constructor " << name << '\n';
    }
    
    ~base_B (){
      std::cout << "Destroying base_B for " << name << '\n';
    }

  private:
    std::string name;
};

class multiple_v1 : public base_B, public base_A
{
  public:
    multiple_v1 (int n, std::string mystr)
      : base_B {mystr}
      , base_A {n}
    {
      std::cout << "Constructing multiple_v1\n";
    }
    ~multiple_v1 (){
      std::cout << "Destroying multiple_v1\n";	
    }

};
class multiple_rev : public base_A, public base_B
{
  public:
    multiple_rev (int n, std::string mystr)
      : base_A {n}
    , base_B {mystr}
    {
      std::cout << "Constructing multiple_rev\n";
    }
    ~multiple_rev (){
      std::cout << "Destroying multiple_rev\n";	
    }

};


int main()
{
  multiple_v1 A {10, "John"};

  std::cout << "\nReverse\n";
  multiple_rev B {20, "Mary"};

  std::cout << "\nEnd of Program\n";
  return 0;
}
