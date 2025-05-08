# Compiler
CXX      = g++

# Flags
CXXFLAGS = -std=c++23 -Wall -Wextra
LDFLAGS  = -ltbb

# Targets
all: assignment6a assignment6b

# Rule for assignment6a (dot product calculation)
assignment6a: assignment6a.cc
	$(CXX) $(CXXFLAGS) -o assignment6a assignment6a.cc $(LDFLAGS)

# Rule for assignment6b (convex hull with dependencies)
assignment6b: assignment6b.cc ConvexHull.cc point.cc
	$(CXX) $(CXXFLAGS) -o assignment6b assignment6b.cc ConvexHull.cc point.cc $(LDFLAGS)

# Clean target
.PHONY: clean
clean:
	$(RM) assignment6a assignment6b *.txt *.png
