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
OBJECTS = main_double.o matrix_ops_double.o matrix_ops_cuda_double.o

TARGET = matrix_reduce_double

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(LINK) $(OBJECTS) -o $(TARGET) $(LIBS)

main_double.o: main_double.c matrix_ops_double.h matrix_ops_cuda_double.h
	$(CC) -c main_double.c $(CFLAGS)

matrix_ops_double.o: matrix_ops_double.c matrix_ops_double.h
	$(CC) -c matrix_ops_double.c $(CFLAGS)

matrix_ops_cuda_double.o: matrix_ops_cuda_double.cu matrix_ops_cuda_double.h
	$(NVCC) -c matrix_ops_cuda_double.cu $(NVCCFLAGS)

clean:
	-$(DEL_FILE) $(OBJECTS) $(TARGET)

# Task 4 - Performance Testing (Double Precision)
test_perf_double: $(TARGET)
	@echo "Running performance tests for Task 4 (Double Precision)..."
	./$(TARGET) -n 1000 -m 1000 -b 4 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 8 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 16 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 32 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 64 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 128 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 256 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 512 -t -o perf_results_double.csv
	./$(TARGET) -n 1000 -m 1000 -b 1024 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 4 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 8 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 16 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 32 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 64 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 128 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 256 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 512 -t -o perf_results_double.csv
	./$(TARGET) -n 5000 -m 5000 -b 1024 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 4 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 8 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 16 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 32 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 64 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 128 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 256 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 512 -t -o perf_results_double.csv
	./$(TARGET) -n 10000 -m 10000 -b 1024 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 4 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 8 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 16 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 32 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 64 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 128 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 256 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 512 -t -o perf_results_double.csv
	./$(TARGET) -n 25000 -m 25000 -b 1024 -t -o perf_results_double.csv
	@echo "Performance tests completed. Results in perf_results_double_double.csv"