# Compilers and commands
CC = gcc
CXX = g++
NVCC = nvcc
LINK = nvcc
DEL_FILE = rm -f

# Flags
CFLAGS = -Wall -O3 -lm
CXXFLAGS = -Wall -O3
NVCCFLAGS = -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75

# Libraries
LIBS = -lm

# Include paths
INCPATH = /usr/include/

# Files
OBJECTS = main.o matrix_ops.o matrix_ops_cuda.o

TARGET = matrix_reduce

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK) $(OBJECTS) -o $(TARGET) $(LIBS)

main.o: main.c matrix_ops.h matrix_ops_cuda.h
	$(CC) -c main.c $(CFLAGS)

matrix_ops.o: matrix_ops.c matrix_ops.h
	$(CC) -c matrix_ops.c $(CFLAGS)

matrix_ops_cuda.o: matrix_ops_cuda.cu matrix_ops_cuda.h
	$(NVCC) -c matrix_ops_cuda.cu $(NVCCFLAGS)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)

# Task 3 - Performance Testing
test_perf: $(TARGET)
	@echo "Running performance tests for Task 3..."
	./$(TARGET) -n 1000 -m 1000 -b 4 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 8 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 16 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 32 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 64 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 128 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 256 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 512 -t -o perf_results.csv
	./$(TARGET) -n 1000 -m 1000 -b 1024 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 4 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 8 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 16 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 32 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 64 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 128 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 256 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 512 -t -o perf_results.csv
	./$(TARGET) -n 5000 -m 5000 -b 1024 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 4 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 8 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 16 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 32 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 64 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 128 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 256 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 512 -t -o perf_results.csv
	./$(TARGET) -n 10000 -m 10000 -b 1024 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 4 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 8 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 16 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 32 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 64 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 128 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 256 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 512 -t -o perf_results.csv
	./$(TARGET) -n 25000 -m 25000 -b 1024 -t -o perf_results.csv
	@echo "Performance tests completed. Results in perf_results.csv"