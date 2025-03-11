/**
 * @file shared_ptr.h
 * @brief Header file for a HPC::shared_ptr for 5614 Assignment 5. 
 * @author R. Morrin
 * @version 1.0
 * @date 2025-03-03
 */
#ifndef SHARED_PTR_H_CJNUWRSH
#define SHARED_PTR_H_CJNUWRSH


#include <iostream>

namespace HPC
{
	template <typename T>
		class shared_ptr
		{
			public:
				shared_ptr() = delete;
				shared_ptr (T*);

				// Move Semantics
				shared_ptr(shared_ptr&&);
				shared_ptr& operator=(shared_ptr&&);

				// Copy Semantics
				shared_ptr(shared_ptr const &);
				shared_ptr& operator=(shared_ptr const &);

				// Destructor
				~shared_ptr ();

				T* get();
				T* release();
				unsigned int use_count();

				T& operator*();
				T operator*() const;

			private:
				T* managed_object;
				unsigned int * counter;
		};

} /* HPC */ 

#endif /* end of include guard: SHARED_PTR_H_CJNUWRSH */
