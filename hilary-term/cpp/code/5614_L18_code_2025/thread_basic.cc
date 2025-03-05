
#include <thread>
#include <iostream>

void f(){
    std::cout << "Function f()" << std::endl;
}

struct Hello
{
    void operator()(){
	std::cout << "Function object" << std::endl;
    }
};

int main()
{
    std::cout << std::thread::hardware_concurrency() << std::endl;
    auto named_lambda = [](){
	std::cout << "Named Lambda"  << std::endl;
    };

    std::thread t1 {f};
    std::thread t2 {Hello()};
    std::thread t3 {named_lambda};
    std::thread t4 {
	[](){std::cout << "Anon lambda" << std::endl;}
    };


    t1.join();
    t2.join();
    t3.join();
    t4.join();
    return 0;
}
