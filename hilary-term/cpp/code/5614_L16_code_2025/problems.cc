#include <memory>
#include <iostream>

int main()
{
   double *A = new double ;
   auto Del = [](double * d){ 
       std::cout << "Custom Deleter \n";
       delete d; 
   };

   std::unique_ptr<double, decltype(Del)> uA {A, Del}; 

    return 0;
}
