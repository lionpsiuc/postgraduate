#include <iostream>
class CPP_class {
    int x=10; 	// Only possible from C++11
    public: 	// Note. Default is private
    int get_x(){  	// Definition inside
	return(x);
    };
    void set_x(int); 	// Declaration
};

// Defining outside body of class
void CPP_class::set_x(int in){
    x = in;
};

int main()
{
    CPP_class A; //create instance of class

    // Cannot access A.x directly now
    //std::cout << "A.x = " << A.x << "\n";
    std::cout << "A.x = " << A.get_x() << "\n";

    A.set_x(1);
    std::cout << "A.x = " << A.get_x() << "\n";

    //A.x = 2; 	// Can not modify A.x directly

    return 0;
}
