#include <iostream>
#include <algorithm>
#include <vector>

double mysq(const double in){
    return in*in;
}

int main()
{
    std::vector<double> v1 {1.1, 2.2, 3.3, 4.4};
    std::vector<double> v1sq (v1.size());

    std::transform(std::cbegin(v1), std::cend(v1), std::begin(v1sq), mysq);

    for(const auto& i: v1sq){
	std::cout << i << '\n';
    }

    return 0;
}
