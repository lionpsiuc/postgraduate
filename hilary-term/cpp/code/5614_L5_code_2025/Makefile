
# Makefile for Lecture 5. 5614
# R. Morrin
# 7 Feb 2025
#


objects := vector3\
	   vector_copy\
	   vector_copy_assign\
	   vector_move\
	   vector_plus\
	   vector_plus_alternate



# Better to use --std=c++23 if you have a newer compiler
CXXFLAGS:= --std=c++23 -Wextra -Wall
#CXXFLAGS:= --std=c++23 -Wextra -Wall -D_GLIBCXX_DEBUG  # Use this to turn on bounds checking for stdvector.cc
CXX 	:= g++

all: $(objects)

$(objects): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean

clean:
	rm -f $(objects)
