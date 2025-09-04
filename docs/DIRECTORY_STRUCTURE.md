# K-means Project Directory Structure

This document describes the organization of the K-means clustering project repository.

## Directory Tree

```
K-means/
├── 📁 analysis_results/          # Generated performance analysis outputs
│   ├── performance plots (PNG)
│   ├── timing data (CSV)
│   └── analysis reports (TXT)
├── 📁 archive/                   # Legacy/backup implementations
│   └── older versions of MPI code
├── 📁 build/                     # Compiled executables
│   ├── KMEANS_seq               # Sequential implementation
│   ├── KMEANS_omp               # OpenMP implementation
│   ├── KMEANS_mpi               # MPI implementation
│   ├── KMEANS_mpi_omp           # MPI+OpenMP hybrid
│   └── KMEANS_cuda              # CUDA GPU implementation
├── 📁 data/                      # Input datasets
│   ├── input*.inp               # Small test datasets (2D-100D)
│   └── *k_100.inp               # Large datasets (200k-800k points)
├── 📁 docs/                      # Documentation
│   ├── implementation guides
│   ├── performance analysis
│   └── usage instructions
├── 📁 logs/                      # Execution and timing logs
│   ├── SLURM job outputs
│   └── timing measurements
├── 📁 results/                   # Clustering output files
│   ├── classification results
│   └── timing files (.timing)
├── 📁 scripts/                   # Automation and analysis tools
│   ├── 📁 analysis/             # Performance analysis scripts
│   │   └── comprehensive_performance_analysis.py
│   ├── 📁 local/                # Local testing scripts
│   ├── 📁 slurm/                # HPC cluster job scripts
│   │   ├── comprehensive_kmeans_test.sh
│   │   └── individual implementation tests
│   └── 📁 utils/                # Utility scripts
├── 📁 src/                       # Source code implementations
│   ├── KMEANS.c                 # Sequential baseline
│   ├── KMEANS_omp.c             # OpenMP parallel
│   ├── KMEANS_mpi.c             # MPI distributed
│   ├── KMEANS_mpi_omp.c         # MPI+OpenMP hybrid
│   └── KMEANS_cuda.cu           # CUDA GPU acceleration
├── 📁 timing_logs/               # Historical timing data
├── Makefile                      # Build configuration
├── README.md                     # Project overview
└── analysis_requirements.txt     # Python dependencies for analysis
```

## Key Components

### 🔧 **Source Implementations** (`src/`)

- **Sequential**: Reference implementation for performance baseline
- **OpenMP**: Shared-memory parallelization (1-64 threads)
- **MPI**: Distributed-memory parallelization (multiple processes)
- **MPI+OpenMP**: Hybrid approach combining both paradigms
- **CUDA**: GPU acceleration for massive parallelism

### 📊 **Performance Analysis** (`scripts/analysis/`)

- Automated performance comparison across all implementations
- Speedup and efficiency analysis
- Scalability studies with various core counts
- Cross-dataset performance evaluation

### 🎯 **Testing Infrastructure** (`scripts/slurm/`)

- Comprehensive test suites for HPC environments
- Automated data collection across multiple configurations
- SLURM job scripts for cluster execution

### 📈 **Results & Analysis** (`analysis_results/`)

- Performance plots and visualizations
- Detailed timing comparisons
- Scaling behavior analysis
- Implementation-specific performance reports

## Usage

1. **Build**: `make` or `make <target>` to compile specific implementations
2. **Test**: Use scripts in `scripts/slurm/` for comprehensive testing
3. **Analyze**: Run `comprehensive_performance_analysis.py` for detailed analysis
4. **Results**: Check `analysis_results/` for generated reports and plots
