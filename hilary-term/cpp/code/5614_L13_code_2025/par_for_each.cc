#include <iostream>
#include <vector>
#include <algorithm>  //Needed for for_each
#include <execution>

void func1(int i){
    std::cout <<  i << '\n';
}

int main()
{
    std::vector<int> v (15);
    std::iota(std::begin(v), std::end(v), 0);

    std::for_each(std::execution::par, std::cbegin(v), std::cend(v), func1);
    return 0;
}
