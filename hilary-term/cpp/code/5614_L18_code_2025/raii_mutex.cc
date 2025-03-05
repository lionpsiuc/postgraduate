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
    std::lock_guard<std::mutex> lck1 {coutmut};
    std::cout << "Hello ";
}

void w(){
    //std::lock_guard<std::mutex> lck2 {coutmut};
    std::scoped_lock lck2 {coutmut};
    std::cout << "World";
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
