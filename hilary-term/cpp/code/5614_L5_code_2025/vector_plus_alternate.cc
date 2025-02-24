/**
 * @file vector2.cc
 * @brief Code for 5614 L4 example
 * 		This is an alternate way of overloading operator+ not as member function of the class.
 * 		This is just to show that it can be done the other way too.
 * @author R. Morrin
 * @version 1.0
 * @date 2023-02-02
 */
#include <iostream>

namespace HPC {
								       //
    class Vector {
	public:
	    Vector() = delete; 
	    Vector (int num) : number_of_elements {num}, data {new double [num]}{
		std::cout << "Constructing vector of size " << number_of_elements << "\n";
		for (auto i = 0; i < num; ++i) {
		    data[i]=0; 
		}
	    };
	    ~Vector (){
		std::cout << "Destroying vector of size " << number_of_elements << "\n";
		delete[] data;
	    };

	    double& operator[](const int i){
		if(i >= number_of_elements){
		    std::cerr << "ERROR: Accesing beyond bounds!!!!" << std::endl;
		}
		return data[i]; 
	    }

	    friend std::ostream& operator<<(std::ostream&, const Vector&);
    	    friend Vector operator+(Vector const & lhs, Vector const & rhs);

	private:
	    int number_of_elements;
	    double * data;
    };

    std::ostream& operator<<(std::ostream& os, const Vector& vec){
	os << "--------\nNumber of elements in vector:\t" << vec.number_of_elements << '\n';
	for (int i = 0; i < vec.number_of_elements; ++i) {
	    os << vec.data[i] << '\t';
	}
	os << std::endl;
	return os;
    }

    Vector operator+(Vector const & lhs, Vector const & rhs){
	    if(lhs.number_of_elements != rhs.number_of_elements){
		    std::cerr << "Mismatch in number of elements\n";
		    exit(1);
	    }
	    Vector result {lhs.number_of_elements};

	    for (int i = 0; i < lhs.number_of_elements; i++) {
		    result[i] = lhs.data[i] + rhs.data[i];
	    }
	    return result;
    }
} /* HPC */ 


	

int main()
{
  HPC::Vector A {5}; 
  HPC::Vector B {5};

  for (auto i = 0; i < 5; ++i) {
	  A[i] = i;
	  B[i] = 3*i+1;
  }

  std::cout << "A:\n" << A <<'\n';
  std::cout << "B:\n" << B <<'\n';

  HPC::Vector C {A+B};
  std::cout << "C:\n" << C << '\n';
  std::cout << "A+C:\n" << A+C << '\n';

  return 0;
}
