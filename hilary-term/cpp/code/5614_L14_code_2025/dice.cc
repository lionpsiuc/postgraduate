#include <iostream>
#include <random>
#include <map>
#include <functional>

int main()
{
    std::default_random_engine de;
    std::uniform_int_distribution<int> one_six {1,6};    
    std::map<int, int> count;

    for (auto i = 1; i <= 6; ++i) {
	count[i]=0;
	// or count.insert(std::make_pair(i,0));
    }

    for (auto i = 0; i < 1e6; ++i) {
       int myran = one_six(de); 
       count[myran]++;
    }

    for(auto const& p : count){
	std::cout << p.first << '\t' << p.second << '\n';
    }
    std::cout << '\n';

    //Use std::bind to create a convenient wrapper
    auto rng = std::bind(one_six, de);
    for (auto i = 0; i < 1e6; ++i) {
       count[rng()]++;
    }

    // C++17 structured bindings for range for loop
    for(const auto& [k,v]  : count){
	std::cout << k << '\t' << v << '\n';
    }

    return 0;
}
