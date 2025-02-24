/**
 * @file function_parameter_passing.cc
 * @brief  Simple function to show the difference between value, reference, and pointer
 * @author R. Morrin
 * @version 1.0
 * @date 2023-02-02
 */
#include <iostream>


/**
 * @brief Dummy function to show effect of passing by value, reference, pointer
 *
 * @param A integer from main
 * @param B reference to an int
 * @param C address of an int
 */
void func_inc(int A, int &B, int *C) {
    A++;
    B++;
    (*C)++;
}

int main() {
   int x = 0, y = 10, z = 20; 
   std::cout << x << "\t" << y << "\t" << z <<"\n";
   func_inc(x,y,&z);
   std::cout << x << "\t" << y << "\t" << z <<"\n";
   return 0;
}

