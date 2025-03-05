#include <cassert>
#include <iostream>

class Myadd
{
public:
    Myadd (int in) : val{in} {};

template <typename T>
	auto operator()(T x) const {
	    return x + val;
	}
private:
    int val;
};



int main()
{
   auto add = Myadd(1);

   // Run time assertion
   assert(add(42) == 43); 

   // Call with double
   std::cout << add(3.2) << '\n';

   // Using generic lambda. Note use of auto in parameters
   auto gen_lamb = [value=1](auto x) {return x+value;};
   assert(gen_lamb(12) == 13);
   std::cout << gen_lamb(4.2) << '\n';

    return 0;
}
