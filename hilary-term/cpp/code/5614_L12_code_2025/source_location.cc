#include <source_location>
#include <iostream>
#include <fstream>
#include <string>


void myopen(const std::string filename){

    std::fstream infile {filename};
    if(infile.is_open()){
	std::cout << "File exists\n";
	infile.close();
	return;
    }

    std::source_location loc {std::source_location::current()};
    std::cerr <<  "Error in " << loc.file_name()
	<< "\nOn line " << loc.line()
	<< "\nOn column " << loc.column()
	<< "\nIn function " << loc.function_name();
}

int main()
{
    const std::string fakefile {"fake.txt"};

    //myopen(fakefile);
    myopen(std::string {"fake.txt"});

    // C way. Preprocessor
    std::cout << "\n\nError in file " << __FILE__ 
	<< "on line " << __LINE__ <<'\n';

    return 0;
}
