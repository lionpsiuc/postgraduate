#include <iostream>

void f1(int x){
   std::cout << "F1 int: " << x << std::endl;
}

void f1(double x){
    std::cout << "F1 double: " << x << std::endl;
}

int main(void)
{
    f1(1);
    f1(1.2);
    f1(static_cast<int> (1.2)); 
    f1(static_cast<double> (1)); 
    return 0;
}
