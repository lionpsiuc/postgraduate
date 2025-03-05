#include <memory>
#include <iostream>

struct Example {
	Example() : value{0}{}
	Example(int n) : value{n}{}

	int value;
};

template <typename T>
struct arr_delete {
	void operator()(const T *p){
		delete[] p;
	}
};

int main()
{
	std::unique_ptr<Example> up {new Example{2}};
	std::unique_ptr<Example[]> up_arr {new Example[10]};
	std::shared_ptr<Example> sp {new Example{1}};
	std::shared_ptr<Example[]> sp_arr {new Example[10]};

	// Older code. Needed custom deleter. But didn't have operator[]
	std::shared_ptr<Example> sp_arr_old {new Example[10], arr_delete<Example>{}};
	std::shared_ptr<Example> sp_arr_old2 {new Example[10], std::default_delete<Example[]>{}};
	std::shared_ptr<Example> sp_arr_old3 {new Example[10], [](Example *e){delete[] e;}};

	// make functions
	auto up_m {std::make_unique<Example>(3)};
	auto up_arr_m {std::make_unique<Example[]>(3)};

	auto sp_m {std::make_shared<Example>(4)};
	// auto sp_arr_m = std::make_shared<Example[]>(5);  // C++20

	
	
	return 0;
}
