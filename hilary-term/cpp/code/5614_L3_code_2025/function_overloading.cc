#include <string>
#include <iostream>
#include <fstream> 	// For writing to file

void ol_print(const double x){
    std::cout << "Printing to stdout " << x << std::endl;
}

void ol_print(const double x, std::string infile){
    std::cout << "Printing to file " << infile << std::endl;
    std::ofstream myfile;  	// myfile is output filestream
    myfile.open (infile.c_str());
    myfile << x << std::endl;
    myfile.close();
}

int main()
{
    ol_print(2.0);
    ol_print(3.0, std::string {"output.dat"});
    return 0;
}
