#include <iostream>
#include <typeinfo>
#include <type_traits> 	   // Needed for enable_if

// concrete function
double sum (double A, double B){
    std::cout << "Concrete function\n";
    return A+B;
}

// templated function Conditional Definition using enable_if!
template <typename T1 
	, typename T2
	, typename = typename std::enable_if<std::is_arithmetic<T1>::value>::type
	, typename = typename std::enable_if_t<std::is_arithmetic_v<T2>>  // C++14/17
	>
// auto sum(T1 A, T2 B) -> decltype(T1{} + T2{}){
auto sum(T1 A, T2 B){
    std::cout << "Templated sum for " << typeid(T1).name() <<
       " & " << typeid(T2).name() << std::endl;
    return A+B;
}

int main()
{

    std::cout << "1+2 = " 	<< sum(1,2) 	<< '\n';
    std::cout << "1.0+2 = " 	<< sum(1.0,2) 	<< '\n';
    std::cout << "1.0+2.0 = " 	<< sum(1.0,2.0) << '\n';
   
//  std::cout << "cat + dog = "	<< sum(std::string("cat"),std::string("dog")) << '\n';
  
//  std::cout << "cat + 3.0 = "	<< sum(std::string("cat"),3.0) << '\n';

    return 0;
}
