#include <iostream>

struct agg_t {
	int x;
	double y;
};


int main()
{
	agg_t X {1, 2.3};

  	std::cout << X.x  << '\t' << X.y << std::endl;	
	return 0;
}
