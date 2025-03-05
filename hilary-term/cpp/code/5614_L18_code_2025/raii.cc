#include <thread>
#include <iostream>

struct raii_thread : std::thread {
    // implicitly inherit std::thread constructors
    using std::thread::thread;
    // bring operator= from std::thread to this scope
    using thread::operator=;
    ~raii_thread(){
	if(joinable()){
	    std::cout << "Joining\n";
	    join();
	}
	std::cout << "Destroying thread!\n";
    }
};



int main()
{
    raii_thread { 
	[](){std::cout << "Running anon lambda!\n";}
    }; 
    return 0;
}
