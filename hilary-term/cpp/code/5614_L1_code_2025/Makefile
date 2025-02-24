# @file Makefile
# @brief  L1 for 5614
# @author R. Morrin
# @version 2.0
# @date 2025-01-31

CC  		:= gcc
CXX 		:= g++
CXXFLAGS 	:=  --std=c++23   # -Wextra -Wall  # Turn off warnings just for this one as I have some examples which will give warnings for this one
LDFLAGS 	:= 


SRCS 		:= auto.cc\
		   initialize.cc\
		   minimal.cc\
		   narrowing.cc

EXES 		:= ${SRCS:.cc=}

all: ${EXES} fast_forward

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


fast_forward: fast_forward.c 
	$(CC) -Wall -Wextra -o $@ $<

.PHONY: clean all

clean:
	$(RM) $(EXES)  fast_forward
