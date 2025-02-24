#include <iostream>

int main()
{
   int a {1};
   int &b {a}; 	// or int &b = a;

   std::cout << "a = " << a 
      << "\tb = " << b << std::endl; 

   b++;
   std::cout << "a = " << a 
      << "\tb = " << b << std::endl; 
    return 0;
}
