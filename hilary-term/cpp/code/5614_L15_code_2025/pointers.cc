
#include <string>
#include <iostream>
#include <fstream>

double sum_data_from_file(const int n, const std::string filename){

    double *array {new double[n]};
    double sum {0};

    std::ifstream infile {filename};
    if(!infile.is_open()){
	throw std::string("Error opening ") + filename;
    }

    for(auto i=0; i<n; ++i){
	infile >> array[i];
	sum+=array[i];
    }

    delete[] array;
    return sum;
}

int main()
{
    try {
	std::cout <<  sum_data_from_file(10, "fakefile.txt") << '\n';
    }
    catch(std::string err){
	std::cout << "Exception caught: " << err <<'\n';

    }catch(...) {
	std::cerr << "Unknown exception\n"; 
    }

    return 0;
}
