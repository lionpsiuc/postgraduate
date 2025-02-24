// Library 2
#include <iostream>
namespace MyLib2
{
    void print_name(){
	std::cout << "This is from library 2\n";
	for (auto i=0; i <10; ++i) {
	    std::cout << i << "\n";
	}
    }

} /* MyLib2 */ 

//Does not have to be all together
namespace MyLib2  
{
    void library2_function() {
	// Do something useful
	//
	return;
    }

} /* MyLib2 */ 
