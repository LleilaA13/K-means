# K-means clustering method
#
# Parallel computing (Degree in Computer Engineering)
# 2022/2023
#
# (c) 2023 Diego Garcia-Alvarez and Arturo Gonzalez-Escribano
# Grupo Trasgo, Universidad de Valladolid (Spain)

# Directory structure
SRCDIR = src
BUILDDIR = build
DATADIR = data
RESULTSDIR = results
LOGSDIR = logs

# Compilers - Auto-detect environment
# Check if we're on macOS (local) or Linux (cluster)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS local development
    CC=clang
    OMPFLAG=-Xpreprocessor -fopenmp -lomp
    LIBOMP_PATH=/opt/homebrew/opt/libomp
    LDFLAGS=-L$(LIBOMP_PATH)/lib
    CPPFLAGS=-I$(LIBOMP_PATH)/include
else
    # Linux cluster environment
    CC=gcc
    OMPFLAG=-fopenmp
    LDFLAGS=
    CPPFLAGS=
endif

MPICC=mpicc
CUDACC=nvcc

# Flags for optimization and libs
FLAGS=-O3 -Wall
LIBS=-lm

# Targets to build
OBJS=$(BUILDDIR)/KMEANS_seq $(BUILDDIR)/KMEANS_omp $(BUILDDIR)/KMEANS_mpi $(BUILDDIR)/KMEANS_mpi_omp $(BUILDDIR)/KMEANS_cuda

# Rules. By default show help
help:
	@echo
	@echo "K-means clustering method"
	@echo
	@echo "Group Trasgo, Universidad de Valladolid (Spain)"
	@echo
	@echo "Directory Structure:"
	@echo "  src/         - Source code files"
	@echo "  build/       - Compiled executables"
	@echo "  data/        - Input test files"
	@echo "  results/     - Output results"
	@echo "  logs/        - Performance logs"
	@echo "  scripts/     - Utility scripts"
	@echo "  docs/        - Documentation"
	@echo
	@echo "Build Targets:"
	@echo "  make KMEANS_seq    Build only the sequential version"
	@echo "  make KMEANS_omp    Build only the OpenMP version"
	@echo "  make KMEANS_mpi    Build only the MPI version"
	@echo "  make KMEANS_mpi_omp Build only the MPI+OpenMP hybrid version"
	@echo "  make KMEANS_cuda   Build only the CUDA version"
	@echo
	@echo "  make all           Build all versions"
	@echo "  make debug         Build with debug info"
	@echo "  make clean         Remove build files and clean results/logs"
	@echo "  make clean-build   Remove only build files"
	@echo "  make clean-results Remove only results directory"
	@echo "  make clean-logs    Remove only logs directory"
	@echo "  make setup         Create directory structure"
	@echo

# Create directory structure
setup:
	@mkdir -p $(SRCDIR) $(BUILDDIR) $(DATADIR) $(RESULTSDIR) $(LOGSDIR) scripts docs

all: setup $(OBJS)

$(BUILDDIR)/KMEANS_seq: $(SRCDIR)/KMEANS.c | $(BUILDDIR)
	$(CC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

$(BUILDDIR)/KMEANS_omp: $(SRCDIR)/KMEANS_omp.c | $(BUILDDIR)
	$(CC) $(FLAGS) $(DEBUG) $(CPPFLAGS) $(LDFLAGS) $(OMPFLAG) $< $(LIBS) -o $@

$(BUILDDIR)/KMEANS_mpi: $(SRCDIR)/KMEANS_mpi.c | $(BUILDDIR)
	$(MPICC) $(FLAGS) $(DEBUG) $< $(LIBS) -o $@

$(BUILDDIR)/KMEANS_mpi_omp: $(SRCDIR)/KMEANS_mpi_omp.c | $(BUILDDIR)
	$(MPICC) $(FLAGS) $(DEBUG) $(OMPFLAG) $< $(LIBS) -o $@

$(BUILDDIR)/KMEANS_cuda: $(SRCDIR)/KMEANS_cuda.cu | $(BUILDDIR)
	$(CUDACC) $(DEBUG) $< $(LIBS) -o $@

# Convenience targets (without path prefix)
KMEANS_seq: $(BUILDDIR)/KMEANS_seq
KMEANS_omp: $(BUILDDIR)/KMEANS_omp  
KMEANS_mpi: $(BUILDDIR)/KMEANS_mpi
KMEANS_mpi_omp: $(BUILDDIR)/KMEANS_mpi_omp
KMEANS_cuda: $(BUILDDIR)/KMEANS_cuda

# Ensure build directory exists
$(BUILDDIR):
	@mkdir -p $(BUILDDIR)


# Remove the build files
clean:
	@echo "Cleaning build, results, and logs directories..."
	rm -rf $(BUILDDIR)/*
	rm -f $(RESULTSDIR)/*
	rm -f $(LOGSDIR)/*
	@echo "✓ Clean complete!"

# Remove only build files
clean-build:
	@echo "Cleaning build directory..."
	rm -rf $(BUILDDIR)/*
	@echo "✓ Build files removed!"

# Remove only results files
clean-results:
	@echo "Cleaning results directory..."
	rm -f $(RESULTSDIR)/*
	@echo "✓ Results directory cleaned!"

# Remove only log files
clean-logs:
	@echo "Cleaning logs directory..."
	rm -f $(LOGSDIR)/*
	@echo "✓ Logs directory cleaned!"

# Compile in debug mode
debug:
	make DEBUG="-DDEBUG -g" FLAGS="-g -Wall" all

# Run performance tests
test-omp: $(BUILDDIR)/KMEANS_omp
	@echo "Running OpenMP performance test..."
	@mkdir -p $(RESULTSDIR) $(LOGSDIR)
	./$(BUILDDIR)/KMEANS_omp $(DATADIR)/input100D.inp 4 100 1 0.001 $(RESULTSDIR)/output_omp.out 42 8

# Show current directory structure
tree:
	@echo "Current repository structure:"
	@find . -type d -name ".git" -prune -o -type d -print | sort

# Show compiler information for debugging
compiler-info:
	@echo "System Information:"
	@echo "  OS: $(UNAME_S)"
	@echo "  CC: $(CC)"
	@echo "  MPICC: $(MPICC)"
	@echo "  OpenMP flags: $(OMPFLAG)"
	@echo "  LDFLAGS: $(LDFLAGS)"
	@echo "  CPPFLAGS: $(CPPFLAGS)"
	@echo ""
	@echo "Compiler availability:"
	@which $(CC) 2>/dev/null && echo "  ✓ $(CC) found" || echo "  ✗ $(CC) not found"
	@which $(MPICC) 2>/dev/null && echo "  ✓ $(MPICC) found" || echo "  ✗ $(MPICC) not found"
	@which $(CUDACC) 2>/dev/null && echo "  ✓ $(CUDACC) found" || echo "  ✗ $(CUDACC) not found"

