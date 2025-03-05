#include <iostream>
#include <memory>

int main()
{
  std::weak_ptr<int> b ;

  { // Create inner scope for a
    auto a = std::make_shared<int>(10);
    b = a;

    /*
     *std::cout << "a is stored at address " << &a <<
     *  " and holds value " << a <<
     *  ". The value pointed to by a is " << *a << '\n';
     */

    // Print number of shared_ptrs holding object. 
    std::cout << "Count :" << b.use_count() << '\n';

    auto sp = b.lock();
    std::cout << "sp is stored at address " << &sp <<
      " and holds value " << sp <<
      ". The value pointed to by sp is " << *sp << '\n';

    std::cout << "Count :" << b.use_count() << '\n';
  }

  std::cout << "Count :" << b.use_count() << '\n';
  if(auto sp = b.lock()){
    std::cout << "sp is stored at address " << &sp <<
      " and holds value " << sp <<
      ". The value pointed to by sp is " << *sp << '\n';
  }

  return 0;
}
