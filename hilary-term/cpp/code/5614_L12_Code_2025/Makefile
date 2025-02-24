# 5614 L12
# R. Morrin
# V 6.0
# 21 Feb 2025

CXX := g++
CXXFLAGS := -W -Wall -std=c++23
LDFLAGS  := 

TARGETS := abbrev_templ\
	   assert\
	   auto_concepts\
	   cin\
	   exception\
	   fin\
	   manip\
	   source_location\
	   stringstream

all: $(TARGETS) instantiation inst_main.o

$(TARGETS): %: %.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

inst_main.o: inst_main.cc instantiation.h
	$(CXX) $(CXXFLAGS)  -c $<

instantiation: instantiation.cc inst_main.o instantiation.h
	$(CXX) $(CXXFLAGS)  -o $@ $^

.PHONY: all clean

clean:
	$(RM) $(TARGETS) *.o
