
#include<cassert>


int main()
{
    // run-time assertion
   assert(1==0);

   //Compile-time assertions
  // static_assert(1==0, "Static_assertion_failed"); 
   //static_assert(1==1, "Static_assertion_failed"); 
    return 0;
}
