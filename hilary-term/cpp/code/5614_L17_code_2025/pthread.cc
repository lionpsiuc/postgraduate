#include <functional>
#include <pthread.h>
#include <array>
#include <iostream>

void* f(void *){
    std::cout << "Function f()" << std::endl;
    return nullptr;
}

void* g(void *){
    std::cout << "Function g()" << std::endl;
    return nullptr;
}

int main()
{
    pthread_t t1 {};
    pthread_t t2 {};


       if(pthread_create(&t1, nullptr, f, nullptr) != 0){
	   	std::cerr << "Error Creating Thread 0\n";
       } 

       if(pthread_create(&t2, nullptr, g, nullptr) != 0){
	   	std::cerr << "Error Creating Thread 0\n";
       } 

       pthread_join(t1, nullptr);
       pthread_join(t2, nullptr);

    return 0;
}
