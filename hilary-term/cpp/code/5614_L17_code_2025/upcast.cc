
struct Parent
{
    Parent (){};
    ~Parent (){};
};

struct Child : public Parent
{
    Child (){};
    ~Child (){};
};


int main()
{
    Child  *ch = new Child;
    Parent *pp = new Parent;

    pp = ch; 	// Ok
		//ch = pp;  // Error. invalid conversion
		// from ‘Parent*’ to ‘Child*’
    delete pp;
    return 0;
}
