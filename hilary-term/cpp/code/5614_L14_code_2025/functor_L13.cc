#include <string>
#include <iostream>

class funct
{
    public:
	funct(std::string str, int x) : name{str}, count{x}{}
	
	void operator()(){
		std::cout << name << ": count = " << count << '\n';
		count++;
	}

    private:
	const std::string name;
	int count;
};


int main()
{

    funct A {"First", 10};
    funct B {"Second", 100};
    
    for (int i = 0; i < 5; ++i) {
       A(); 
    }

    for (int i = 0; i < 5; ++i) {
       B(); 
    }

    return 0;
}
