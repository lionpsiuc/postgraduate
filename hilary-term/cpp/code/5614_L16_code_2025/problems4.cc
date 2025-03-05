#include <memory>
#include <iostream>

int main()
{
   auto Del = [](auto * d){ 
       std::cout << "Custom Deleter \n";
       delete[] d; 
   };
   // Now this time with shared_ptrs
   std::shared_ptr<double> sA {new double, Del}; 

   // Now copy construct sA2 from sA
   std::shared_ptr<double> sA2 {sA}; 

    return 0;
}
