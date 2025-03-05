#include <iostream>
#include <thread>
#include <mutex> 	// Needed for mutex
#include <memory>
#include <chrono>

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
std::mutex incmut;

auto sec = std::chrono::seconds(1);

void h(std::shared_ptr<int> i){
    coutmut.lock();
    std::this_thread::sleep_for(sec);
    incmut.lock();
    std::cout << "Hello " << (*i)++;
    incmut.unlock();
    coutmut.unlock();
}

void w(std::shared_ptr<int> i){
    incmut.lock();
    std::this_thread::sleep_for(sec);
    coutmut.lock();
    std::cout << "World" << (*i)++;
    coutmut.unlock();
    incmut.unlock();
}

int main()
{
    auto n = std::make_shared<int>(0);
    raii_thread t1 {h, n}; 
    raii_thread t2 {w, n}; 

    return 0;
}
