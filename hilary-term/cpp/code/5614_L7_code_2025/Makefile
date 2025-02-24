#
# Makefile for Lecture 7. 5614
# R. Morrin
# 12 Feb 2025
#


objects :=  derived\
	    friend\
	    friend_func\
	    multiple\
	    singleton\
	    singleton_meyers
	    # 5614_L7_question



CXXFLAGS := --std=c++23 -Wextra -Wall
CXX 	:= g++

all: $(objects)

$(objects): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<


.PHONY: clean

clean:
	rm $(objects)
