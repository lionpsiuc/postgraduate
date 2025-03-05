#include <memory>
#include <iostream>

int main()
{
   double *A = new double ;
   auto Del = [](auto * d){ 
       std::cout << "Custom Deleter \n";
       delete d; 
   };

   std::unique_ptr<double, decltype(Del)> uA {A, Del}; 

   // This will compile without warning! But will give undefined behaviour
   std::unique_ptr<double, decltype(Del)> uA2 {A, Del}; 

    return 0;
}
