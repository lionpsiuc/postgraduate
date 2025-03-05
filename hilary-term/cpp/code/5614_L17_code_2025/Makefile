# @file Makefile
# @brief  Lecture 17 for 5614
# 	Note, not all these programs need to be linked against libtbb.so or have the -fopenmp flag
# 	but I wrote it to compile all the same way just to minimize work.
# @author R. Morrin
# @version 3.0
# @date 2025-03-03

CXX 		:= g++
CXXFLAGS 	:= -Wextra -Wall --std=c++23
LDFLAGS 	:= -ltbb  -fopenmp


SRCS 		:= c_cast.cc\
		   const_cast.cc\
		   dynamic_cast.cc\
		   dynamic_cast_smart_pointers.cc\
		   execution_policy.cc\
		   fork.cc\
		   openmp.cc\
		   pthread.cc\
		   ssize.cc\
		   upcast.cc\
		   vevox_question.cc





EXES 		:= ${SRCS:.cc=}

all: ${EXES}

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


.PHONY: clean all

clean:
	$(RM) $(EXES)
