#include <iostream>
#include <thread>
#include <mutex> 	// Needed for mutex

struct raii_thread : std::thread {
    using std::thread::thread;
    using thread::operator=;
    ~raii_thread(){
	if(joinable()){
	    join();
	}
    }
};

std::mutex coutmut;

void h(){
    coutmut.lock();
	std::cout << "Hello ";
}

void w(){
    coutmut.lock();
	std::cout << "World";
    coutmut.unlock();
}

int main()
{
    raii_thread t1 {h}; 
    raii_thread t2 {w}; 

    std::thread t3 {h};
    std::thread t4 {w};
    t3.join();
    t4.join();

    return 0;
}
