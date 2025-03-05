# Makefile for  5614 L14
#
# Author 	R. Morrin
#
# Modified 	2025-02-25
#

CXX 		:= g++
CXXFLAGS 	:= -Wextra -Wall --std=c++23


SRCS 		:= dice.cc\
		   functor_L13.cc\
		   lambda1.cc\
		   lambda2.cc\
		   lambda_capture.cc\
		   lambda_generic.cc\
		   mt_seeded.cc\
		   random_device.cc\
		   random_gaussian.cc\
		   ranges.cc\
		   ranges2.cc\
		   replace_if.cc\
		   sort_lambda.cc\
		   valarrays2.cc\
		   valarrays_bm.cc\
		   valarrays.cc



EXES 		:= ${SRCS:.cc=}

all: ${EXES}

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean all

clean:
	$(RM) $(EXES)
