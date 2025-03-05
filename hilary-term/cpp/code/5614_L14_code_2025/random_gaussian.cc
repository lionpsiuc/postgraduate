#include <random>
#include <iostream>
#include <functional>
#include <map>

int main()
{
    const int num_sims {10000000};
    const double stepsize {0.1};
    std::mt19937_64  RNG_Engine {};

    //Bind Normal probability density function to Mersenne Twister in order to generate Gaussian numbers.
    auto gen = std::bind(std::normal_distribution<double>{0,1.0},RNG_Engine);

    std::map<double, int> hist;

    for (auto i = 0; i < num_sims; ++i) {
	double ran = gen();
	int sign = ran < 0 ? -1 : 1;
	double index = (static_cast<int>(ran/stepsize + 0.5*sign))*stepsize;
	hist[index]++;
    }

    for(auto const& p : hist){
	double density = p.second/static_cast<double>(num_sims)/stepsize;
	std::cout << p.first << '\t' << density << std::endl;
    }

    return 0;
}
