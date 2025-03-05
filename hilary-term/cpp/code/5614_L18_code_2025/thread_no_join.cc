#include <thread>
#include <iostream>

int main()
{
	std::thread t1{ [](){
		std::cout << "Inside thread\n";	
	}};
	return 0;
}
