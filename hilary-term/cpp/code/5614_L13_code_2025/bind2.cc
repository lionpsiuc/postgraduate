#include <iostream>
#include <cassert>
#include <functional>

double mypow(const int n, double val){
  assert(n>=0); // Check for 
  double res=1;
  for (auto i = 0; i < n; ++i) {
    res*=val; 
  }
  return res;
}


int main()
{
  auto sq = std::bind(mypow, 2, std::placeholders::_1);
  std::cout << sq(3.0) << '\n';

  // alias std::placeholders::_3 to _3
  using std::placeholders::_3;
  auto cube = bind(mypow, 3, _3);
  std::cout << cube(2.0, 3.0, 4.0) << '\n';

  return 0;
}
