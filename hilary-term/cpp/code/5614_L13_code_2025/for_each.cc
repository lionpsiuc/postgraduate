#include <iostream>
#include <vector>
#include <algorithm>  //Needed for for_each

void func1(int i){
    std::cout <<  i << '\n';
}

int main()
{
    std::vector<int> v {5,4,3,2,1};
    std::for_each(std::cbegin(v), std::cend(v), func1);
    return 0;
}
