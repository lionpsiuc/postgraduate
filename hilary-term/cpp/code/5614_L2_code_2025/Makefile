# @file Makefile
# @brief  L2 for 5614
# @author R. Morrin
# @version 2.0
# @date 2025-01-31

CXX 		:= g++
CXXFLAGS 	:=  --std=c++23   # -Wextra -Wall  # Turn off warnings just for this one as I have some examples which will give warnings for this one
LDFLAGS 	:= 


SRCS 		:= array.cc\
		   dynamic_alloc.cc\
		   references.cc\
		   references_vs_pointer.cc\
		   scope.cc\
		   using_declaration.cc\
		   using_directive.cc\
		   using_directive_common.cc\
		   void_pointer.cc
		   # scope1.cc 	# This one gives error deliberately


EXES 		:= ${SRCS:.cc=}

all: ${EXES} library_main library_main_ns

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

library_main: library_main.cc libl1.so libl2.so
	$(CXX) -o $@  $< $(CXXFLAGS)  -ll1 -ll2 -L.

libl1.so: library1.cc
	$(CXX) --shared -o $@  $<  -fPIC $(CXXFLAGS)

libl2.so: library2.cc
	$(CXX) --shared -o $@  $<  -fPIC $(CXXFLAGS)

library_main_ns: library_main_ns.cc libl1_ns.so libl2_ns.so
	$(CXX) -o $@  $< $(CXXFLAGS)  -ll1_ns -ll2_ns -L.

libl1_ns.so: library1_ns.cc
	$(CXX) --shared -o $@  $<  -fPIC $(CXXFLAGS)

libl2_ns.so: library2_ns.cc
	$(CXX) --shared -o $@  $<  -fPIC $(CXXFLAGS)


.PHONY: clean all

clean:
	$(RM) $(EXES)  libl1.so libl2.so libl1_ns.so libl2_ns.so library_main library_main_ns
