#include <iostream>

int main()
{
   int a {1};
   int *p {&a}; // or int *p = &a;

   std::cout << "a = " << a 
      << "\t*p = " << *p << std::endl; 

   (*p)++;
   std::cout << "a = " << a 
      << "\t*p = " << *p << std::endl; 
    return 0;
}
