/**
 * @file non_templated.cc
 * @brief For Assignment 3 of 5614.
 * 	Copy main function to templated.cc and then write a templated function at the top which can replace all four function definitions below.
 * @author R. Morrin
 * @version 1.0
 * @date 2025-02-25
 */
#include <iostream>
#include <cstddef>




/**
 * @brief Function to compute the dot product of two vectors stored as C-style arrays
 * 		Note that normally, I would be making better use of const here.
 * @param[in] x pointer to first array
 * @param[in] y pointer to second array
 * @param[in] N Number of points in each array
 *
 * @return  The dot product of x.y.
 */
int dot(int *x, int *y, int N){
	int result {0};
	for (int i = 0; i < N; ++i) {
		result += x[i] * y[i];		
	}
	return result;
}

double dot(double *x, int *y, int N){
	double result {0};
	for (int i = 0; i < N; ++i) {
		result += x[i] * y[i];		
	}
	return result;
}

double dot(int *x, double *y, int N){
	double result {0};
	for (int i = 0; i < N; ++i) {
		result += x[i] * y[i];		
	}
	return result;
}

double dot(double *x, double *y, int N){
	double result {0};
	for (int i = 0; i < N; ++i) {
		result += x[i] * y[i];		
	}
	return result;
}

int main()
{
	int n       {4};
	double A[]  {1,2,3,4};
	int B[]     {5,6,7,8};
	
	std::cout << "A.A = " << dot(A,A,n) 
		<< "\nB.B = " << dot(B,B,n)
		<< "\nA.B = " << dot(A,B,n)
		<< "\nB.A = " << dot(B,A,n) << '\n';

	return 0;
}
