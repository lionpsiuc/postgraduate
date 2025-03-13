# Compiler
CXX := g++
CXXFLAGS := -std=c++23 -Wextra -Wall

# Target executable
TARGET := assignment3

# Object files
OBJS := portfolio.o assignment3.o

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to build portfolio.o
portfolio.o: portfolio.cc portfolio.h instruments.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Rule to build assignment3.o
assignment3.o: assignment3.cc instruments.h portfolio.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean target
.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)
