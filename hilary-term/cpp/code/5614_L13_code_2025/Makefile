# Makefile for L13 for 5614
# Author R. Morrin
# V 3.0
# 2025-02-25

# Note: I also have code for parallel for_each that needs an additional library
# Your machine may be different
# g++ --std=c++23 -o par_for_each par_for_each.cc -ltbb
#
# Also, you will need t least c++20 for ranges
# g++ -o ranges ranges.cc --std=c++20

CXX 		:= g++
CXXFLAGS 	:= -Wextra -Wall --std=c++23 
LDFLAGS 	:=  


SRCS 		:= accumulate.cc\
		   all_of.cc\
		   bind2.cc\
		   bind.cc\
		   for_each.cc\
		   functor.cc\
		   functor_templated.cc\
		   sort.cc\
		   transform.cc



EXES 		:= ${SRCS:.cc=} 

all: ${EXES} par_for_each def_main_v2  # def_main   # Remove first comment to see compilation errors

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $< 

par_for_each: par_for_each.cc
	$(CXX) $(CXXFLAGS) -ltbb -o $@ $< 

def_main: def_main.o lib.o
	$(CXX) $(LDFLAGS) -o $@ $^

def_main.o: def_main.cc def.h
	$(CXX) $(CXXFLAGS) -c $<

lib.o: lib.cc def.h
	$(CXX) $(CXXFLAGS) -c $<

def_main_v2: def_main_v2.o lib_v2.o
	$(CXX) $(LDFLAGS) -o $@ $^

def_main_v2.o: def_main_v2.cc def_v2.h
	$(CXX) $(CXXFLAGS) -c $<

lib_v2.o: lib_v2.cc def_v2.h
	$(CXX) $(CXXFLAGS) -c $<



.PHONY: clean all

clean:
	$(RM) $(EXES) par_for_each def_main  def_main.o lib.o  def_main_v2  def_main_v2.o lib_v2.o
