#include <iostream>

double * mat_vec3(const double mat[][3], const double *const vec){

    double *result {new double[3]}; // dynamically allocate memory

    for (int i = 0; i < 3; i++) {
	result[i] = 0;
	for (int j = 0; j < 3; ++j) {
	    result[i] += mat[i][j]*vec[j]; 
	}
    }
    return(result);
}

int main()
{
    const double A[][3] = {{1.0, 2.0, 3.0},
	{4.0, 5.0, 6.0},
	{7.0,8.0,9.0}};	

    double B[] = {1.0, 2.0, 3.0}; 

    double *res = mat_vec3(A,B);
    for (int i = 0; i < 3; ++i) {
	std::cout << res[i] << "\t";
    }
    std::cout << std::endl;

    delete[] res; // "free" memory
    return 0;
}
