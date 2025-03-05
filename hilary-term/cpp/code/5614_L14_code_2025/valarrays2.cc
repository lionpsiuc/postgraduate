#include <iostream>
#include <iomanip>
#include <valarray>
#include <iterator>
#include <algorithm>

int main()
{
    std::valarray<double> v1 (5);
    std::valarray<double> v2 (5);
    std::valarray<double> res_add (5);
    std::valarray<double> res_mul (5);
    std::valarray<double> res_exp (5);

    // Populate the valarrys
    std::generate(std::begin(v1), std::end(v1), [i=0]() mutable {return i++;});
    std::generate(std::begin(v2), std::end(v2), [i=9]() mutable {return i--;});

    // Perform numerical operations on valarrays
    res_add = v1 + v2;
    res_mul = v1 * v2;
    res_exp = std::exp(v1);

    // print results to screen
    std::cout << std::setw(2) << "v1"
	<< std::setw(6) << "v2" 
	<< std::setw(8) << "v1+v2"
	<< std::setw(7) << "v1*v2"
	<< std::setw(9) << "exp(v1)\n"
	<< std::setprecision (3);
    for (auto i = 0U; i < v1.size(); ++i) {
    std::cout << std::setw(2) << v1[i]
	<< std::setw(6) << v2[i] 
	<< std::setw(6) << res_add[i]
	<< std::setw(6) << res_mul[i]
	<< std::setw(6) << res_exp[i] << '\n';
    }

    return 0;
}
