#include <iostream>
void f2(int a, int b=0);
void f2(int a);

int main()
{
    f2(1, 2); 	// Fine
    f2(3); 	// Error. Ambiguous
    return 0;
}

void f2(int x, int y){
    std::cout << "x =" << x
	<< " y = " << y << "\n";
}
void f2(int x){
    std::cout << " x =" << x << "\n";
}
