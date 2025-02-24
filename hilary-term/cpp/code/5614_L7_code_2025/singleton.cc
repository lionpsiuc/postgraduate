#include <iostream>

class basic_singleton
{
    public:
	static basic_singleton* get_instance(){
	    if(instance == nullptr){
		std::cout << "First call" << std::endl;
		instance = new basic_singleton;
	    }	
	    else{
		++count;
		std::cout << "Times called " << count << std::endl;
	    }
	    return instance;
	}
	basic_singleton (const basic_singleton&) = delete;
	basic_singleton& operator=(const basic_singleton&) = delete;
    private:
	basic_singleton() = default;
	static basic_singleton* instance;
	static int count;  // Needs to be static to be used inside static function above.
};

// Initialise static member variables
basic_singleton* basic_singleton::instance {nullptr};
int basic_singleton::count {0};

int main()
{
    //basic_singleton A {}; // ERROR. Cannot call constructor
    basic_singleton* A {basic_singleton::get_instance()};
    basic_singleton* B {basic_singleton::get_instance()};
    basic_singleton* C {basic_singleton::get_instance()};
    basic_singleton* D {basic_singleton::get_instance()};
    //basic_singleton E {*D}; 	// Deleted

    std::cout << A << '\n' << B <<  '\n' << C << '\n' << D << '\n';
    return 0;
}
