XX      = g++
MPICXX   = mpicxx

COMMON_FLAGS = -std=c++23 -Wall -Wextra
CXXFLAGS     = $(COMMON_FLAGS)
LDFLAGS      = -ltbb
MPI_CXXFLAGS = $(COMMON_FLAGS) -DMPI
MPI_LIBS     = -lboost_mpi -lboost_serialization

all: assignment6a assignment6b assignment6c

assignment6a: assignment6a.cc
	$(CXX) $(CXXFLAGS) -o assignment6a assignment6a.cc $(LDFLAGS)

assignment6b: assignment6b.cc ConvexHull.cc point.cc
	$(CXX) $(CXXFLAGS) -o assignment6b assignment6b.cc ConvexHull.cc point.cc $(LDFLAGS)

assignment6c:
	$(MPICXX) $(MPI_CXXFLAGS) -o assignment6c assignment6c.cc ConvexHull.cc point.cc $(MPI_LIBS)

.PHONY: clean
clean:
	$(RM) assignment6a assignment6b assignment6c *.txt *.png
