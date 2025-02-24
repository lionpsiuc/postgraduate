
#include <typeinfo>
#include <iostream>
#include "instantiation.h"
namespace HPC
{

    template <typename T>
	Instantiation<T>::Instantiation(T in) : X{in} {
	    std::cout << "Instantiating for type " << typeid(T).name() << std::endl;
	};

    // Need explicit instantiations here
    // Code will only be generated for explicitly instantiated types
    template class Instantiation<double>;
    
} /* HPC */ 
