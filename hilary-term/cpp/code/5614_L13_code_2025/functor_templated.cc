#include <string>
#include <iostream>

template <typename T>
class Less_than
{
  public:
    Less_than (const T& in) : val{in} {};
    bool operator()(const T& x) const {
      return x<val;
    }
    T get_val(){return val;};

  private:
    const T val;
};


int main()
{
  Less_than<int>         lti {50};
  Less_than<double>      ltd {0.5};
  Less_than<std::string> lts {"mystring"};

  auto ival={10, 100, 30 ,55, 76, 21};
  auto dval={0.25, 0.55, 0.75, -0.4, 0.1};
  auto sval={ "abc", "ABC", "b", "xyz"};

  for (auto x : ival) { 		// Note generic code
    if(lti(x)) { 			// Calling functor
      std::cout << x << " is less than " << lti.get_val() << '\n';
    }
  }

  for (const auto& x : dval) { 		// const auto ref
    if(ltd(x)) {
      std::cout << x << " is less than " << ltd.get_val() << '\n';
    }
  }

  for (const auto&x : sval) {
    if(lts(x)) {
      std::cout << x << " is less than " << lts.get_val() << '\n';
    }
  }

  return 0;
}
