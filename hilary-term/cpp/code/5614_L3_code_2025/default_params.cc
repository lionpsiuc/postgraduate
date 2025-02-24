#include <iostream>	
// Function with default parameters.
int mymult(int A=1, int B=2, int C=3){
    return(A*B*C);
}

int main()
{
    std::cout << "1:\t" << mymult(10,10,10) << "\n"
     << "2:\t" << mymult(10,10) <<  "\n"
     << "3:\t" << mymult(10) <<  "\n"
     << "4:\t" << mymult() << "\n";
    return 0;
}

