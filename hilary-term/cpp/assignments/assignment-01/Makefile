# Compiler
CXX = g++
CXXFLAGS = -std=c++23 -Wall -Wextra

# Flags
LDFLAGS =

# Source and object files
SRCS = $(wildcard *.cc)
EXES = $(SRCS:.cc=)

# Default target
all: $(EXES)

# Rule to build each executable
$(EXES): %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Clean target
.PHONY: clean
clean:
	$(RM) $(EXES) x-positive.txt y-even.txt
