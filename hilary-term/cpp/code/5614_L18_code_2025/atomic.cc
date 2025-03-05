#include <iostream>
#include <atomic>
#include <thread>


int main()
{
    const int N {10'000};
    std::atomic<int> atom {0};
    int n {0};

    auto lamda = [&]{
	for(auto i=0; i<N; ++i){
	    atom++;
	    n++;
	}
    };

    std::thread t1{lamda};
    std::thread t2{lamda};

    t1.join();
    t2.join();

    std::cout << "n = " << n <<
	"\natom = " << atom << '\n';

    return 0;
}
