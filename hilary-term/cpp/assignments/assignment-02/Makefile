CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra
LDFLAGS =

SRCS = $(wildcard *.cc)

EXES = $(SRCS:.cc=)

all: $(EXES)

$(EXES): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(EXES)
