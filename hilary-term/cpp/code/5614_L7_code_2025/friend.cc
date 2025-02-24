#include <iostream>
class myclass
{
public:
    myclass (int in) : a {in}{};
    // Declare fclass as a friend
    friend class fclass;

private:
    int a;
};

class fclass
{
public:
    fclass (){};
    // fclass can access private members of myclass
    void print_myclass(myclass & in){
	std::cout << "myclass::a = " << in.a << '\n';
    }
};

int main()
{
   myclass A {10};
   fclass B;

   B.print_myclass(A);

    return 0;
}
