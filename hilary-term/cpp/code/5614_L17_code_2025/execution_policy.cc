#include <functional>
#include <iostream>
#include <chrono>
#include <valarray>
#include <vector>
#include <numeric>
#include <random>
#include <execution>

std::default_random_engine eng{};

int main()
{
    using namespace std::chrono;
    const int n {100'000'000};
    double sum {0};

    auto ui = std::bind(std::uniform_real_distribution<>{0,1}, eng);

    std::vector<double> vec(n);
    std::vector<double> res(n);
    std::valarray<double> val(n);
    for (int j = 0; j < n; ++j) {
	vec[j] = ui();
	val[j] = vec[j];
    }

    auto start = steady_clock::now();
    for (int j = 0; j < n; ++j) {
	sum += vec[j];
    }
    auto end = steady_clock::now();

    std::cout << "Vector loop: sum = " << sum << "\nElapsed time : " 
	<< duration_cast<milliseconds>(end - start).count() << " ms\n\n";

    start = steady_clock::now();
    double sum2 {std::accumulate(vec.begin(), vec.end(),0.0)};
    end = steady_clock::now();
    std::cout <<  "vector::accumulate: sum2 = " << sum2 << "\nElapsed time : " 
	<< duration_cast<milliseconds>(end - start).count() << " ms\n\n";


    start = steady_clock::now();
    double sum3 {val.sum()};
    end = steady_clock::now();
    std::cout <<  "Valarry: sum3 = " << sum3 <<  "\nElapsed time : " 
	<< duration_cast<milliseconds>(end - start).count() << " ms\n\n";

    start = steady_clock::now();
    double sum4 {std::reduce(std::execution::par_unseq, vec.begin(), vec.end())};
    end = steady_clock::now();
    std::cout <<   "Parallel reduce: sum4 = " << sum4 <<  "\nElapsed time : " 
	<< duration_cast<milliseconds>(end - start).count() << " ms\n\n";


    start = steady_clock::now();
    std::transform(vec.begin(), vec.end(), res.begin(), [](double v){return std::sin(v) + std::log(v);});
    end = steady_clock::now();
    std::cout <<   "Serial Transform = " << res[0] << "\nElapsed time : " 
	<< duration_cast<milliseconds>(end - start).count() << " ms\n\n";

    start = steady_clock::now();
    std::transform(std::execution::par_unseq, vec.begin(), vec.end(), res.begin(), [](double v){return std::sin(v) + std::log(v);});
    end = steady_clock::now();
    std::cout <<  "Parallel Transform = " << res[0] << "\nElapsed time : " 
	<< duration_cast<milliseconds>(end - start).count() << " ms\n\n";
    return 0;
}

