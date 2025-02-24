#
# Makefile for L10. 5614
#
# Updated: 19/02/2025
#

CXX := g++
CFLAGS := -W -Wall -std=c++23


TARGETS := ctad\
	   specialisation\
	   specialised\
	   template2\
	   template3\
	   template4\
	   template5\
	   template_ol\
	   unordered_map\
	   vec1\
	   vec2\
	   vector_templated\


all: $(TARGETS)

$(TARGETS): %: %.cc
	$(CXX) $(CFLAGS) -o $@ $<


.PHONY: all clean

clean:
	$(RM) $(TARGETS)
