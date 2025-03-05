#include <memory>
#include <iostream>

int main()
{
   double *A = new double;
   auto Del = [](auto * d){ 
       std::cout << "Custom Deleter \n";
       delete d; 
   };
   // Now this time with shared_ptrs
   std::shared_ptr<double> sA {A, Del}; 

   // This will compile without warning! But will give undefined behaviour
   std::shared_ptr<double> sA2 {A, Del}; 

    return 0;
}
