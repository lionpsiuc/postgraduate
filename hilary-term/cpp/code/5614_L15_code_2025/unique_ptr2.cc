#include <memory>
#include <iostream>

class MyArr
{
    public:
	MyArr (int n): sz{n}, data{new int[n]}{};
	~MyArr (){ std::cout << "MyArr Dtor for size " << sz << "\n";};
	int sz;

    private:
	int *data;
};

struct D1 {
    void operator()(MyArr *p){
	std::cout << "Deleter #1 for size " << p->sz << '\n';
	//delete p;  // Create mem leak
    }
};

struct D2 {
    void operator()(MyArr *p){
	std::cout << "Deleter #2 for size " << p->sz << '\n';
	delete p;  
    }
};

int main()
{

    std::unique_ptr<MyArr, D1> p1(new MyArr {1}, D1{});
    std::unique_ptr<MyArr, D2> p2(new MyArr {2}, D2{});
    std::unique_ptr<MyArr, D2> p3(new MyArr {3}, D2{});
    /* Below is also fine as D1/D2 default constructible
    std::unique_ptr<MyArr, D1> p1(new MyArr {1});
    std::unique_ptr<MyArr, D2> p2(new MyArr {2});
    std::unique_ptr<MyArr, D2> p3(new MyArr {3});
    */

    auto p4 = std::move(p1);

    std::unique_ptr<MyArr> p5;
    p5.reset(p2.release()); 	// Doesn't forward the deleter!

    return 0;
}
