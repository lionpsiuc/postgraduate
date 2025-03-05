#include <thread>
#include <iostream>

int main()
{
	std::jthread t1{ [](){
		std::cout << "Inside thread\n";	
	}};
	return 0;
}
