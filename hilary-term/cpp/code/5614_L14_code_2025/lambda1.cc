#include <iostream>
#include <vector> 	// needed for vector
#include <algorithm> 	// Needed for for_each

// Class which has operator() overloaded so that
// if will print a particular value if its index is
// a multiple of m

class mod_print
{
  public:
    //constructor
    mod_print (std::ostream& s, int in): os{s}, m{in} {};
    //overloading operator()
    void operator()(int x) const {
      if (x%m==0) {
	os << x <<'\n'; 
      }
    }

  private:
    std::ostream& os;
    int m;
};


int main()
{
  std::vector<int> v {1, 2, 3, 4, 5}; 

  std::ostream& mycout {std::cout};
  int mod {2};

  // Using function object
  std::cout << "functor\n";
  std::for_each(v.begin(), v.end(), mod_print{mycout,mod});
  // std::ranges::for_each(v, mod_print{mycout,mod}); // Using ranges

  std::cout << "\nlambda\n";
  // Using lamda expression
  std::for_each(v.begin(), v.end(), [&mycout,mod](int x){if (x%mod==0) mycout << x <<'\n';});

  return 0;
}
