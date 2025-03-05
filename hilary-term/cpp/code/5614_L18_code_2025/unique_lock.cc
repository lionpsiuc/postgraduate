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
    std::unique_lock<std::mutex> l1 {coutmut, std::defer_lock};
    std::this_thread::sleep_for(sec);
    std::unique_lock<std::mutex> l2 {incmut, std::defer_lock};
    std::lock(l1,l2);
    std::cout << "Hello " << (*i)++;
}

void w(std::shared_ptr<int> i){
    std::unique_lock<std::mutex> l3 {incmut, std::defer_lock};
    std::this_thread::sleep_for(sec);
    std::unique_lock<std::mutex> l4 {coutmut, std::defer_lock};
    std::lock(l3,l4);
    std::cout << "World" << (*i)++;
}

int main()
{
    auto n = std::make_shared<int>(0);
    raii_thread t1 {h, n}; 
    raii_thread t2 {w, n}; 

    return 0;
}
