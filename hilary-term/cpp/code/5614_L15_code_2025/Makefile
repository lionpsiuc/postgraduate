# @file Makefile
# @brief  Makefile for 5614 L15
# @author R. Morrin
# @version 3.0
# @date 2025-02-27

CXX 		:= g++
CXXFLAGS 	:= -Wextra -Wall --std=c++20
LDFLAGS 	:=

SRCS 		:= custom_deleter.cc\
		   pointers.cc\
		   unique_ptr2.cc\
		   unique_ptr.cc


EXES 		:= ${SRCS:.cc=}

all: ${EXES} 

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<



.PHONY: clean all

clean:
	$(RM) $(EXES) 
