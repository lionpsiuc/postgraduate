#
# Makefile for Lecture 8. 5614
# R. Morrin
# 14 Feb 2025
#


objects :=  derived2 derived3 derived4 diamond diamond_virtual 

CXXFLAGS:= -Wextra -Wall --std=c++23
CXX 	:= g++

all: $(objects)

$(objects): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean

clean:
	rm -f $(objects)
