
int main()
{
    int a{1}; //define integer "a" and initialise it to value "1"

    {
	double b{2};
	a++; 	// Fine
	b++; 	// Fine
    }
    a++; 	// Fine
    b++; 	// ERROR. b is not in scope here
    return 0;
}
