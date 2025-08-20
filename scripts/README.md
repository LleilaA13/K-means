# K-means Scripts Directory

This directory contains all scripts for building, running, and analyzing K-means implementations across different platforms and environments.

## Directory Structure

```
scripts/
├── README.md                 # This file - overview of all scripts
├── local/                    # Local execution scripts
│   ├── run_kmeans.sh        # General-purpose K-means runner
│   └── cluster_run.sh       # Legacy cluster runner (deprecated)
├── slurm/                    # SLURM cluster scripts
│   ├── README.md            # SLURM-specific documentation
│   ├── slurm_omp_performance.sh    # OpenMP scaling test
│   ├── slurm_comprehensive.sh      # Multi-version comparison
│   ├── slurm_cuda.sh               # GPU performance test
│   ├── slurm_python.sh             # Python analysis runner
│   └── cluster_analysis.py         # Comprehensive analysis tool
├── analysis/                 # Analysis and visualization tools
│   ├── log_module.py        # Performance analysis and logging
│   ├── compare_module.py    # Results comparison tool
│   ├── plot2d_module.py     # 2D visualization for small datasets
│   └── quick_test.py        # Fast testing utility
└── utils/                    # Utility scripts (empty, for future use)
```

## Quick Start Guide

### Local Development

```bash
# Run basic performance analysis
./local/run_kmeans.sh

# Analyze performance logs
python3 analysis/log_module.py

# Compare different implementations
python3 analysis/compare_module.py
```

### SLURM Cluster (Sapienza University HPC)

```bash
# Quick OpenMP test
sbatch slurm/slurm_omp_performance.sh

# Comprehensive analysis
sbatch slurm/slurm_python.sh

# Check results
cat ../logs/cluster_performance_report.txt
```

## Script Categories

### 🖥️ Local Execution (`local/`)

Scripts for running K-means on local machines with various configurations.

- **`run_kmeans.sh`**: Universal runner for all K-means versions
  - Automatically detects available implementations
  - Supports OpenMP, MPI, CUDA, and sequential versions
  - Configurable datasets and parameters

### ⚡ SLURM Cluster (`slurm/`)

Production-ready scripts for HPC cluster environments.

- **`slurm_omp_performance.sh`**: OpenMP scaling analysis (1-8 threads)
- **`slurm_comprehensive.sh`**: Full comparison (Sequential, OpenMP, MPI)
- **`slurm_cuda.sh`**: GPU vs CPU performance testing
- **`slurm_python.sh`**: Automated analysis with comprehensive reporting
- **`cluster_analysis.py`**: Advanced Python tool for cluster environments

### 📊 Analysis Tools (`analysis/`)

Data analysis, visualization, and performance evaluation tools.

- **`log_module.py`**:

  - Performance analysis with speedup calculations
  - Automatic timing log processing
  - Multi-version performance comparison
  - Matplotlib visualization support

- **`compare_module.py`**:

  - Cross-implementation result validation
  - Clustering quality comparison
  - Error detection and reporting

- **`plot2d_module.py`**:

  - 2D dataset visualization
  - Cluster boundary plotting
  - Before/after comparison plots

- **`quick_test.py`**:
  - Fast correctness validation
  - Small dataset testing
  - Development workflow support

## Usage Patterns

### Development Workflow

1. **Initial Testing**: `analysis/quick_test.py`
2. **Performance Analysis**: `analysis/log_module.py`
3. **Result Validation**: `analysis/compare_module.py`
4. **Visualization**: `analysis/plot2d_module.py`

### Production Workflow

1. **Local Validation**: `local/run_kmeans.sh`
2. **Cluster Testing**: `slurm/slurm_omp_performance.sh`
3. **Full Analysis**: `slurm/slurm_python.sh`
4. **Result Review**: Check generated reports

### Research Workflow

1. **Comprehensive Testing**: `slurm/slurm_comprehensive.sh`
2. **GPU Evaluation**: `slurm/slurm_cuda.sh`
3. **Data Analysis**: `analysis/log_module.py`
4. **Publication Plots**: `analysis/plot2d_module.py`

## Environment Requirements

### Local Scripts

- **macOS/Linux**: bash/zsh shell
- **Compilers**: gcc with OpenMP, optional MPI and CUDA
- **Python**: 3.6+ with numpy, matplotlib (for analysis tools)

### SLURM Scripts

- **HPC Cluster**: SLURM workload manager
- **Partition**: students (24h, 8 cores) or department_only (72h, unlimited)
- **Modules**: gcc, optional MPI/CUDA modules
- **Python**: Available system installation

## Common Commands

### Check Script Status

```bash
# List all scripts
find scripts/ -name "*.sh" -o -name "*.py" | sort

# Check executable permissions
ls -la scripts/*/*.{sh,py}

# Verify directory structure
tree scripts/
```

### Make Scripts Executable

```bash
chmod +x scripts/*/*.sh scripts/*/*.py
```

### Monitor Cluster Jobs

```bash
# Check job queue
squeue -u $USER

# Monitor specific job
watch -n 30 squeue -j <job_id>

# Check job history
sacct -u $USER --starttime=today
```

## Best Practices

1. **Always test locally** before submitting to cluster
2. **Use appropriate resource requests** for SLURM jobs
3. **Check logs directory** for output files
4. **Validate results** with comparison tools
5. **Document parameters** when running experiments

## Troubleshooting

### Common Issues

- **Permission denied**: Run `chmod +x script_name.sh`
- **Module not found**: Load required modules or check PATH
- **Python import errors**: Install required packages or use cluster modules
- **SLURM job failures**: Check resource limits and queue policies

### Debug Mode

Add to any script for verbose output:

```bash
set -x  # Enable debug mode
set +x  # Disable debug mode
```

## Contributing

When adding new scripts:

1. Place in appropriate subdirectory
2. Follow naming conventions
3. Update this README
4. Add proper documentation headers
5. Test on target platform
