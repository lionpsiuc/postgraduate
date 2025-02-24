#include <iostream>
#include <array>

std::array<double, 3> mat_vec3(const std::array<const std::array<double, 3>, 3>& mat,
	const std::array<double,3>& vec){

    std::array<double, 3> result;

    for(auto i=0L; i < std::ssize(mat); ++i){
	result[i] = 0;
	for(auto j = 0LU; j< mat[i].size(); ++j){
	    result[i] += mat[i][j] * vec[j];
	}
    }
    return(result);
}

int main()
{
    const std::array<const std::array<double,3>,3> A {{{1.0, 2.0, 3.0},
	{4.0, 5.0, 6.0},
	{7.0,8.0,9.0}}};	

    const std::array<double,3> B {1.0, 2.0, 3.0}; 

    //double *res = mat_vec3(A,B);

    for(auto &i : mat_vec3(A,B)){
	std::cout << i << '\t';
    }
    std::cout << std::endl;

    return 0;
}
