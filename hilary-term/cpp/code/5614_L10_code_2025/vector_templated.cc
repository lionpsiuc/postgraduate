#include <iostream>
#include <typeinfo>

namespace HPC
{
    template <typename T>
	class Vector
	{
	    public:
		Vector (int num) : number_of_elements {num}, data {new T [num]}{
		    std::cout << "Constructing vector of type " << typeid(T).name() << '\n';
		};
		~Vector (){
		    std::cout << "Destroying vector of type " << typeid(T).name() << '\n';
		    delete[] data;
		};

		T get_element_by_idx(int idx){
		    return data[idx];
		}

		Vector() = delete; // Explicit note we intended not to have default
	    private:
		int number_of_elements;
		T * data;
	};
} /* HPC */ 

int main()
{
    HPC::Vector<std::string> A  {3}; // Create vector of strings
    HPC::Vector<double> B  {5} ;     // Create vector of doubles
    return 0;
}
