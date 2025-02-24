#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>

double mysqrt(double in){
    if(in < 0){
	throw -1;
    }
    return sqrt(in);
}

double get_elem_sq(std::vector<int> & in, unsigned int idx){

    if(idx >= in.size() ){
	std::stringstream os;
	os << "Error. Trying to access index " << idx 
	    << " although vector size is " << in.size() ;
	throw os.str().c_str();
    }
    return in[idx]*in[idx];
}

double get_elem_sq2(std::vector<int> & in, unsigned int idx){
    return in.at(idx)*in.at(idx);
}

void my_alloc(double **in, size_t size){
    *in = new double [size];
}

// Not meant to throw an exception. NOTE noexcept keyword
void my_alloc2(double **in, size_t size) noexcept{
    *in = new double [size];
}

int main()
{

    std::vector<int> v {1,2,3,4};
    double *A = nullptr;

    try {

	// Will cause throw and catch of an int.
	// mysqrt(-10);

	// Will cause throw and catch of const char *
	//get_elem_sq(v,10);
	// get_elem_sq2(v,10);
	

	// new should fail
	// my_alloc(&A, 1e10);
	 my_alloc2(&A, 1e10);
    
    }catch(const char* msg) {
	std::cerr <<"Error: Caught message " << msg  << '\n';
    }
    catch(int n) {
	std::cerr << "Error: Caught integer " << n  << '\n';
    }
    catch(const std::exception& ex){
	std::cout << "Error allocating meory!!!!!\n";
	std::cout << "Caught " << ex.what() <<'\n';
    }
    catch(...){
	std::cerr << "Error: Caught unspecified error\n";
    }

    std::cout << "Program continuing ....\n";
    
    return 0;
}
