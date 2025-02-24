/**
 * @file vector2.cc
 * @brief Code for 5614 L5 example
 * @author R. Morrin
 * @version 1.0
 * @date 2022-02-11
 */
#include <iostream>

namespace HPC {
  class Vector {
    public:
      Vector() = delete; 
      Vector (int num);
      ~Vector ();

      double& operator[](const int i);
      friend std::ostream& operator<<(std::ostream&, const Vector&);
      Vector operator+(const Vector&);

      double length() const;
      int get_num_elems() const {return number_of_elements;}

    private:
      const int number_of_elements;
      double * data;
  };

  double Vector::length() const{
    double mag {0};
    for (int i = 0; i < number_of_elements; i++) {
      mag += data[i] * data[i]; 
    }
    return mag;
  }


  std::ostream& operator<<(std::ostream& os, const Vector& vec){
    os << "--------\nNumber of elements in vector:\t" << vec.number_of_elements << '\n';
    for (int i = 0; i < vec.number_of_elements; ++i) {
      os << vec.data[i] << '\t';
    }
    os << std::endl;
    return os;
  }



  Vector::~Vector (){
    std::cout << "Destroying vector of size " << number_of_elements << "\n";
    delete[] data;
  };

  Vector::Vector (int num) : number_of_elements {num}, data {new double [num]}{
    std::cout << "Constructing vector of size " << number_of_elements << "\n";
    for (auto i = 0; i < num; ++i) {
      data[i]=0; 
    }
  };

  double& Vector::operator[](const int i){
    if(i >= number_of_elements){
      std::cerr << "ERROR: Accesing beyond bounds!!!!" << std::endl;
    }
    return data[i]; 
  }

  Vector Vector::operator+(const Vector& rhs){
    if(number_of_elements != rhs.number_of_elements){
      std::cerr << "Mismatch in number of elements\n";
      exit(1);
    }
    Vector result {number_of_elements};

    for (int i = 0; i < number_of_elements; i++) {
      result[i] = data[i] + rhs.data[i];
    }
    return result;
  }
} /* HPC */ 

int main()
{
  HPC::Vector A {5}; 

  for (auto i = 0; i < 5; ++i) {
	  A[i] = i;
  }

  std::cout << "Length of A = " << A.length() <<'\n';

  return 0;
}
