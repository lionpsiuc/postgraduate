/**
 * @file matrix.h
 * @brief Templated Matrix class for Assignment 4 of 5614 C++ programming
 * @author R. Morrin
 * @version 3.0
 * @date 2025-02-25
 */

#ifndef MATRIX_H_FIYNRKOJ
#define MATRIX_H_FIYNRKOJ

#include <iostream>
#include <iomanip>
#include <fstream>
#include "hpc_concepts.h"
#include "logging.h"

namespace HPC
{
	
	/**
	 * @brief Class to hold a Matrix. 5614 Assignment 3
	 *  	Modified from https://isocpp.org/wiki/faq/operator-overloading#matrix-subscript-op
	 *  	Store the matrix data as a single C-style array
	 *  	Students need to write function definitions for all the included member functions (except the deleted constructor)
	 *
	 * @tparam T Number type template parameter
	 */
	template <Number T>
		class Matrix {
			public:
				Matrix() = delete; 								// Delete default constructor. Really only a comment.
				Matrix(std::size_t const rows, std::size_t const cols); 			// Overloaded Constructor
				Matrix(std::string const file); 						// Overloaded Constructor. Read data from file.
				T& operator() (std::size_t const row_num, std::size_t const col_num);        	// non-const version which can be used to modify objects
				T  operator() (std::size_t const row_num, std::size_t const col_num) const;  	// Const version to be called on const objects

				~Matrix();                              // Destructor
				Matrix(Matrix const& m);               	// Copy constructor
				Matrix& operator=(Matrix const& m);   	// Assignment operator
				Matrix(Matrix && in); 			// Move Constructor
				Matrix& operator=(Matrix&& in); 	// Move Assignment operator

				// Note: Need the template brackets here in declaration.
				friend std::ostream& operator<< <>(std::ostream& os, Matrix<T> const& in);

				std::size_t get_num_rows() const{ return rows;};
				std::size_t get_num_cols() const{ return cols;};

			private:
				std::size_t rows, cols;
				T* data;
		};


	/**
	 * @brief Basic overloaded constructor for Matrix<T> class
	 * 		Note that this will set the values in data to zero.
	 * 		Don't need to check for negative values as size_t will be unsigned.
	 *
	 * @tparam T 	Number type that the matrix contains
	 * @param[in] num_rows Number of rows in created matrix
	 * @param[in] num_cols Number of columns in created matrix.
	 */
	template <Number T>
		Matrix<T>::Matrix(std::size_t const num_rows, std::size_t const num_cols)
		: rows {num_rows}
		, cols {num_cols}
		, data {new T [rows * cols]{}}
	{
		std::cout << "Constructing " << rows << " x " << cols << " Matrix\n";
	}

} /*  HPC */ 
#endif /* end of include guard: MATRIX_H_FIYNRKOJ */
