#
# Makefile for Lecture 3. 5614
# R. Morrin
# 31 Jan 2025
#


objects :=  default_params fol foreach for_loops function_overloading function_parameter_passing more_generic passing_array_to_fn passing_std_array_to_fn vargs const_constexpr consteval_fn constexpr_fn

# Excluded as will fail to compile by design
#overloading_vs_default 


# Better to use --std=c++23 if you have a newer compiler
CXXFLAGS:= --std=c++23 -Wextra -Wall
CXX 	:= g++

all: $(objects)

$(objects): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean

clean:
	rm -f $(objects)
