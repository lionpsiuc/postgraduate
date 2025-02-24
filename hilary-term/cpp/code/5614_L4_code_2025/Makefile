#
# Makefile for Lecture 4. 5614
# R. Morrin
# 7 Feb 2025
#


objects := agg\
	   class\
	   stdvector\
	   struct\
	   struct_constructor\
	   struct_constructor_destructor\
	   vector1\
	   vector2




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
