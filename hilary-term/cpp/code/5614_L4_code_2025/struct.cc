#include <iostream>
struct CPP_struct {
    int x=10; 	// Only possible from C++11
    int get_x(){  	// Definition inside
	return(x);
    };
    void set_x(int); 	// Declaration
};

// Defining outside body of struct
void CPP_struct::set_x(int in){
    x = in;
};

int main()
{
    CPP_struct A; //create instance of structure

    // Can access A.x two ways now.
    std::cout << "A.x = " << A.x << "\n";
    std::cout << "A.x = " << A.get_x() << "\n";

    A.set_x(1);
    std::cout << "A.x = " << A.get_x() << "\n";

    A.x = 2; 	// Can modify A.x directly
    std::cout << "A.x = " << A.get_x() << "\n";

    // Create a CPP_struct on heap
    CPP_struct *B = new CPP_struct;
    B->set_x(100);
    std::cout << "\nB->x = " << B->get_x() << "\n";

    return 0;
}
