#include <iostream>
#include <thread>

struct raii_thread : std::thread {
    using std::thread::thread;
    using thread::operator=;
    ~raii_thread(){
	if(joinable()){
	    join();
	}
    }
};

void h(){
	std::cout << "Hello " 
	 << std::flush << "from h() ";
}

void w(){
	std::cout << "World " 
	 << std::flush << "from w() ";
}

int main()
{
    raii_thread t1 {h}; 
    raii_thread t2 {w}; 

    // use jthread for t3
    std::jthread t3 {h};
    std::thread t4 {w};
    t4.join(); // Manual join for t4

    return 0;
}
