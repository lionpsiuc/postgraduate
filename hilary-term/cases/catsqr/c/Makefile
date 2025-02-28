CC = mpicc
CFLAGS = -O3 -std=c2x -Wall -Wextra
LDFLAGS = -fopenmp -lblas -llapack -llapacke -lm -lmpi -lopenblas

SRCS = $(wildcard *.c)
EXES = $(SRCS:.c=)

all: $(EXES)

$(EXES): %: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	$(RM) $(EXES) *.png *.txt
