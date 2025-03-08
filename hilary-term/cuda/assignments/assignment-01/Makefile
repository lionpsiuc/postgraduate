# Compilers
CC = gcc
NVCC = nvcc

# Flags
CFLAGS = -std=c2x -Wall -Wextra -O3 -lm -D_POSIX_C_SOURCE=199309L -D_GNU_SOURCE
NVCCFLAGS = -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75

# Directories
SINGLEDIR = .
DOUBLEDIR = ./double
OBJDIR = obj

# Source files
SINGLE_C = $(wildcard $(SINGLEDIR)/*.c)
DOUBLE_C = $(wildcard $(DOUBLEDIR)/*.c)
SINGLE_CUDA = $(wildcard $(SINGLEDIR)/*.cu)
DOUBLE_CUDA = $(wildcard $(DOUBLEDIR)/*.cu)

# Object files
SINGLE_OBJS_C = $(patsubst $(SINGLEDIR)/%.c, $(OBJDIR)/%.o, $(SINGLE_C))
DOUBLE_OBJS_C = $(patsubst $(DOUBLEDIR)/%.c, $(OBJDIR)/%.o, $(DOUBLE_C))
SINGLE_OBJS_CUDA = $(patsubst $(SINGLEDIR)/%.cu, $(OBJDIR)/%.o, $(SINGLE_CUDA))
DOUBLE_OBJS_CUDA = $(patsubst $(DOUBLEDIR)/%.cu, $(OBJDIR)/%.o, $(DOUBLE_CUDA))

# Combined object files for each target
SINGLE_OBJS = $(SINGLE_OBJS_C) $(SINGLE_OBJS_CUDA)
DOUBLE_OBJS = $(DOUBLE_OBJS_C) $(DOUBLE_OBJS_CUDA)

# Executable targets
SINGLE_TARGET = main-single
DOUBLE_TARGET = main-double

# Default target
all: $(SINGLE_TARGET) $(DOUBLE_TARGET)

# Link the final executables
$(SINGLE_TARGET): $(SINGLE_OBJS) | $(OBJDIR)
	$(NVCC) $(SINGLE_OBJS) -o $@
$(DOUBLE_TARGET): $(DOUBLE_OBJS) | $(OBJDIR)
	$(NVCC) $(DOUBLE_OBJS) -o $@

# Compile C source files
$(OBJDIR)/%.o: $(SINGLEDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@
$(OBJDIR)/%.o: $(DOUBLEDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files
$(OBJDIR)/%.o: $(SINGLEDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
$(OBJDIR)/%.o: $(DOUBLEDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Ensure directory exists
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Clean target
.PHONY: clean
clean:
	$(RM) $(SINGLE_TARGET) $(DOUBLE_TARGET) $(SINGLE_OBJS) $(DOUBLE_OBJS) *.csv
	rm -rf $(OBJDIR) plots

# For the third task
.PHONY: task3
task3: $(SINGLE_TARGET)
	@for N in 1000 5000 10000 25000; do \
		for B in 4 8 16 32 64 128 256 512 1024; do \
			echo "\nRunning with matrix size = $$N x $$N and number of threads per block = $$B..."; \
			echo "----------------------------------------------------------------------------------"; \
			./$(SINGLE_TARGET) -n $$N -m $$N -b $$B -t -o results-task3.csv; \
		done; \
	done
	@echo "\nResults saved in results-task3.csv"

# For the fourth task
.PHONY: task4
task4: $(DOUBLE_TARGET)
	@for N in 1000 5000 10000 25000; do \
			echo "\nRunning with matrix size $$N x $$N and number of threads per block = 256..."; \
			echo "-------------------------------------------------------------------------------"; \
			./$(DOUBLE_TARGET) -n $$N -m $$N -b 256 -t -o results-task4.csv; \
		done
	@echo "\nResults saved in results-task4.csv"
