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
    myBase *md  {new myDerived};
    myBase *md2 {new myDerived2};

    // ERROR: error: invalid initialization of reference of type ‘const myDerived2&’ from expression of type ‘myBase’
    //fn_for_myDerived2_only(*md)
    //fn_for_myDerived2_only(*md2)

    if(auto mdr = dynamic_cast<myDerived2 *>(md)){
	std::cout << "First if\n";
	fn_for_myDerived2_only(*mdr);
    }
    if(auto mdr2 = dynamic_cast<myDerived2 *>(md2)){
	std::cout << "Second if\n";
	fn_for_myDerived2_only(*mdr2);
    }

    return 0;
}
