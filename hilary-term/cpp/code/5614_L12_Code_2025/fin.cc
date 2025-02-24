#include <iostream>
#include <fstream>

int main()
{
    std::string name;
    int age;

    std::cout << "Please enter a name and age\n";
    std::cin >> name >> age;;

    std::cout << "Writing to file\n";
    std::ofstream outfile;
    outfile.open("output.txt");
    outfile << name <<" is " << age << " years old\n";
    outfile.close();

    std::cout << "\nNow reading from file\n";
    std::ifstream infile;
    infile.open("output.txt");
    if (! infile.is_open()) { // Should always check!
	    std::cerr << "Could not open output.txt\n";
	    return -1;
    }
    std::string line;
    while(infile){
	getline(infile, line);
	std::cout << line ;
    }
    std::cout << '\n';
    return 0;
}

