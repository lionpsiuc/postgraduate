#include <iostream>
#include <chrono>
#include <thread>


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
  int count {0};

  auto f1 = [&count](int max){
    for(auto i =0; i<max; ++i){
      ++count;
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  };

  raii_thread t1{f1, 3};

  auto pr = [&count, &t1]{
    while(t1.joinable()){
      std::cout << count << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  };

  raii_thread t2{pr};
  t1.join();

  return 0;
}
