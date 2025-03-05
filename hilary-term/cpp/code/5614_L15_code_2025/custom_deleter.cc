#include <iostream>
#include <cassert>
#include <vector>
#include <memory> 	// Needed for unique_ptr.

struct Closer{
    void operator()(std::FILE *fp) const {
	if(fp == nullptr){ exit(EXIT_FAILURE);};
	std::cout << "Closing File" << std::endl;
	std::fclose(fp);
    }

};

void write_to_file(const std::vector<int>& in){

    std::FILE *fp = std::fopen("vect.txt", "w");
    std::unique_ptr<std::FILE, Closer> uptr{fp};
    assert(in.size());
    // Write to file etc.
    //
    
    // File will automatically be closed here
    std::cout << "About to exit function" << std::endl;
}


int main()
{
    std::vector<int> vec {1,2,3,4,5};
    std::cout << "Before calling Function"<< std::endl;
    write_to_file(vec); 
    std::cout << "After calling Function\n";
    return 0;
}
