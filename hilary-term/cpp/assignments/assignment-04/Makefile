# Compiler
CXX := g++
CXXFLAGS := -std=c++20 -Wall -Werror -Wextra -Wsign-conversion

# Target executable
TARGET := assignment4

# Object files
OBJS := assignment4.o logging.o matrix-operations.o

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to build assignment4.o
assignment4.o: assignment4.cc hpc-concepts.h logging.h matrix.h matrix-operations.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Rule to build logging.o
logging.o: logging.cc logging.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Rule to build matrix-operations.o
matrix-operations.o: matrix-operations.cc hpc-concepts.h logging.h matrix.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

ctad: ctad.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

is-floating: is-floating.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

templated: templated.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

try-catch: try-catch.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

# Clean target
.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS) ctad is-floating templated try-catch
