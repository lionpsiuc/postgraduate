#include <initializer_list>
#include <memory>
#include <vector>
#include <iostream>

int main()
{

    auto params_list = {10, 20};

    // The below does not work. "Too many arguments"
    // auto ptr_to_vec = std::make_shared<std::vector<int>>({10, 20});
    
    // This is fine
    auto ptr_to_vec = std::make_shared<std::vector<int>>(params_list);

    for ( auto &i : *ptr_to_vec){
	std::cout << i << '\t';
    }
    std::cout << '\n';

    return 0;
}
