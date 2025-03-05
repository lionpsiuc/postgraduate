#include <random>
#include <iostream>
#include <algorithm>
#include <functional>

int main()
{
    std::vector<unsigned int> random_data(std::mt19937_64::state_size);
    std::cout << "Creating vector with " << std::mt19937_64::state_size << " elements\n";

    std::random_device source{};
    std::generate(std::begin(random_data), std::end(random_data), std::ref(source));
    std::seed_seq seeds(std::begin(random_data), std::end(random_data));
    std::mt19937_64 seededEngine {seeds};

    std::uniform_int_distribution<int> dist {1, 6};

    for (auto i = 0; i < 5; ++i) {
       std::cout << dist(seededEngine) <<'\n';
    }

    return 0;
}
