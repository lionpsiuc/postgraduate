#include <sstream>
#include <iostream>

int main()
{
  std::stringstream mystr("5614 C++ Programming. HPC MSc");
  std::stringstream outstr;
  std::string word;

  int count=0;
  while(mystr>>word){   // Read into "word"
        count++;
        outstr << count <<":\t" << word << '\n';
  }

  // Note the .str() member .function
  std::cout << outstr.str() << '\n';
  return 0;
}

