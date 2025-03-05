#include <string>
#include <iostream>
#include <memory>

class myBase
{
    public:
	myBase (){};
	virtual ~myBase (){};
};

class myDerived : public myBase
{
    public:
	myDerived (){};
	~myDerived (){};
};

class myDerived2 : public myBase
{
    public:
	myDerived2 (){};
	~myDerived2 (){};
};

void fn_for_myDerived2_only(const myDerived2 &){
    std::cout << "In fn_for_myDerived2_only\n";
}

int main()
{
    std::shared_ptr<myBase> md {std::make_shared<myDerived>()};
    std::shared_ptr<myBase> md2 {std::make_shared<myDerived2>()};

    // ERROR: error: invalid initialization of reference of type ‘const myDerived2&’ from expression of type ‘myBase’
    // fn_for_myDerived2_only(*md)
    // fn_for_myDerived_only(*md2)
    
    if(auto mdr = std::dynamic_pointer_cast<myDerived2, myBase>(md)){
	std::cout << "HERE" << '\n';
	fn_for_myDerived2_only(*mdr);
    }
    if(auto mdr2 = std::dynamic_pointer_cast<myDerived2, myBase>(md2)){
	std::cout << "HERE 2" << '\n';
	fn_for_myDerived2_only(*mdr2);
    }
    
    return 0;
}
