#include <iostream>

int main()
{
   int A[3]  {1, 2, 3}; 	// A[0]=1, A[1]=2, A[2]=3
   int B[]   {4, 5, 6}; 	// B[0]=4, B[1]=5, B[2]=6
   int C[3]  {7, 8}; 		// C[0]=7, C[1]=8, C[2]=0
   //int D[2] = {9, 8, 7}; 	// Error. Too many initialisers

   for (auto i = 0; i < 3; ++i) {
    std::cout << "A[" << i <<"]="<<A[i] << "\t"
    << "B[" << i <<"]="<<B[i] << "\t"
    << "C[" << i <<"]="<<C[i] << std::endl;
   }
    return 0;
}
