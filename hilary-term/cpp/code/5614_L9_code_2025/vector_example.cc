#include <iostream>
#include <vector>

struct Example {
	Example(double in) : value{in} {}
	void say_value(){
		std::cout << "Value is " << value << '\n';
	}
	double value;
};


int main(void)
{
	std::vector<double> vecdoub (4);

	for(auto& i: vecdoub){
		i = drand48();
	}

	for(int j =0; j<4; j++){
		std::cout << vecdoub[j] << ' ';
	}
	std::cout << "\n\n";

	Example A {1.2};
	Example B {3.4};

	std::vector<Example *> vecex;

	vecex.push_back(&A);
	vecex.push_back(&B);

	for(const auto& k : vecex){
		k->say_value();
	}

	return 0;
}
