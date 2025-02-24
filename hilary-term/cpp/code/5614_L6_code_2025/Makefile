#
# Makefile for Lecture 6. 5614
# R. Morrin
# 16 Feb 2024
#




objects := mutable vector_free_fns vector_member_fns vector_move vector_swap vector_swap_move

CFLAGS 	:= --std=c++17 -Wextra -Wall
CC 	:= g++

all: $(objects)

$(objects): %: %.cc
	$(CC) $(CFLAGS) -o $@ $<


.PHONY: clean

clean:
	rm -f $(objects)
