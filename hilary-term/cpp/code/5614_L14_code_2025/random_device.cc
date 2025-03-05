
#include <iostream>
#include <string>
#include <map>
#include <random>
 
// Modified from https://en.cppreference.com/w/cpp/numeric/random/random_device
int main()
{
    std::random_device rd;
    std::map<int, int> hist;
    std::uniform_int_distribution<int> dist{0, 5};
    for (int n = 0; n < 10000; ++n) {
        ++hist[dist(rd)]; // note: demo only: the performance of many 
                          // implementations of random_device degrades sharply
                          // once the entropy pool is exhausted. For practical use
                          // random_device is generally only used to seed 
                          // a PRNG such as mt19937
    }
    for (const auto& p : hist) {
        std::cout << p.first << " : " << std::string(p.second/100, '*') << '\n';
    }
return 0;
}
