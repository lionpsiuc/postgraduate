# Compiler
CC = mpicc
CFLAGS = -std=c2x -Wall -Wextra

# Flags
LDFLAGS = -fopenmp -lblas -llapack -llapacke -lm -lmpi -lopenblas

# Source and object files
SRCS = $(wildcard *.c)
EXES = $(SRCS:.c=)

# Default target
all: $(EXES)

# Rule to build each executable
$(EXES): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Clean target
.PHONY: clean
clean:
	$(RM) $(EXES) *.png *.txt
