#include <iostream>
class myclass
{
    public:
	myclass (int in) : a {in}{};
	// Declare fclass as a friend
	friend void print_mc(myclass &);

    private:
	int a;
};

void print_mc(myclass & in){
    std::cout << in.a << '\n';
}

int main()
{
    myclass A {10};

    print_mc(A);

    return 0;
}
