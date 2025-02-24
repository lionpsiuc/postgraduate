// Based on https://www.modernescpp.com/index.php/creational-patterns-singleton#

#include <iostream>

class meyers_singleton{

    private:
	meyers_singleton() = default;
	~meyers_singleton() = default;

    public:
	// Delete copy and move sematics.
	meyers_singleton(const meyers_singleton&) 		= delete;
	meyers_singleton& operator = (const meyers_singleton&) 	= delete;
	meyers_singleton(meyers_singleton&&) 			= delete;
	meyers_singleton& operator = (meyers_singleton&&)	= delete;

	static meyers_singleton& getInstance(){
	    static meyers_singleton instance;        // Create local static member
	    return instance;
	}
};


int main() {

    std::cout << "&meyers_singleton::getInstance(): "<< &meyers_singleton::getInstance() << '\n';
    std::cout << "&meyers_singleton::getInstance(): "<< &meyers_singleton::getInstance() << '\n';

}
