#
# Makefile for Lecture 9. 5614
# R. Morrin
# 12 Feb 2025
#


objects :=   vector_move vector_example no_template template template2 template3 template4 template5 template_ol # move 

CXXFLAGS:= -Wextra -Wall --std=c++23
CXX 	:= g++

all: $(objects)

$(objects): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean

clean:
	rm -f $(objects)
