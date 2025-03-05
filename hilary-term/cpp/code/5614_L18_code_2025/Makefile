# @file Makefile
# @brief  L18 for 5614
# @author R. Morrin
# @version 3.0
# @date 2025-03-03

CXX 		:= g++
CXXFLAGS 	:= -Wextra -Wall --std=c++23 
LDFLAGS 	:= -ltbb 


SRCS 		:= atomic.cc\
		   deadlock2.cc\
		   deadlock_simple.cc\
		   jthread.cc\
		   mutex.cc\
		   no_mutex.cc\
		   raii.cc\
		   raii_mutex.cc\
		   sharing_mem.cc\
		   thread_basic.cc\
		   thread_local.cc\
		   thread_no_join.cc\
		   thread_shared.cc\
		   unique_lock.cc



EXES 		:= ${SRCS:.cc=}

all: ${EXES}

${EXES}: %: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)


.PHONY: clean all

clean:
	$(RM) $(EXES)
