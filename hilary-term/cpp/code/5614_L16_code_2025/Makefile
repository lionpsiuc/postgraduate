# @file Makefile
# @brief  Makefile for L16 for 5614
# @author R. Morrin
# @version 5.0
# @date 2025-03-03


CXX 		:= g++
CXXFLAGS 	:= -Wextra -Wall --std=c++23


SRCS 		:= all_smart_pointers.cc\
		   braced_make.cc\
		   circular1.cc\
		   circular2.cc\
		   circular3.cc\
		   circular4.cc\
		   dangling.cc\
		   dangling_weak.cc\
		   make_shared.cc\
		   problems2.cc\
		   problems3.cc\
		   problems4.cc\
		   problems.cc\
		   sp_arrays.cc




EXES 		:= ${SRCS:.cc=}

all: ${EXES}

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean all

clean:
	$(RM) $(EXES)
