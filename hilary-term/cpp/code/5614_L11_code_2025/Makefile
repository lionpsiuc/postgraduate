#
# Description: Makefile for L11 5614
# Author: rmorrin
# Date:  21-Feb-2025

CXX := g++
CXXFLAGS := -W -Wall -std=c++23

TARGETS := auto_return_type_deduction\
	   decltype\
	   iterator\
	   sfinae\
	   template2\
	   template_errs\
	   traits\
	   traits1

all: $(TARGETS) 

$(TARGETS): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<



.PHONY: all clean

clean:
	$(RM) $(TARGETS) 
