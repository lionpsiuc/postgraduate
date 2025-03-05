#include "def.h"
#include <iostream>


void f3(int x){
	std::cout << "f3\n";
	f2(x*x);
}
