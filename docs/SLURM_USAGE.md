# K-means SLURM Cluster Usage Guide

## Overview

This directory contains SLURM job scripts designed for the Sapienza University HPC cluster. Each script is optimized for different testing scenarios within the cluster constraints.

## Cluster Specifications (Students Partition)

- **Partition**: `students`
- **Walltime**: 24 hours maximum
- **CPUs**: 8 cores maximum per job
- **Memory**: 8-32GB available
- **GPU**: RTX Quadro 6000 (when requested with `--gres=gpu:1`)

## Available Scripts

### 1. Basic OpenMP Performance Test

**File**: `slurm_omp_performance.sh`
**Purpose**: Tests OpenMP scaling from 1-8 threads with detailed performance metrics
**Usage**:

```bash
sbatch scripts/slurm_omp_performance.sh
```

**Output**: `logs/slurm_performance_results.txt`

### 2. Comprehensive Multi-Version Analysis

**File**: `slurm_comprehensive.sh`
**Purpose**: Compares Sequential, OpenMP, and MPI across multiple datasets
**Usage**:

```bash
sbatch scripts/slurm_comprehensive.sh
```

**Output**: `logs/slurm_comprehensive_results.txt`

### 3. CUDA GPU Performance Test

**File**: `slurm_cuda.sh`
**Purpose**: Tests CUDA version against CPU implementations
**Requirements**: GPU access (`--gres=gpu:1`)
**Usage**:

```bash
sbatch scripts/slurm_cuda.sh
```

**Output**: `logs/slurm_cuda_results.txt`

### 4. Python-Based Analysis Tool

**File**: `slurm_python.sh` + `cluster_analysis.py`
**Purpose**: Comprehensive analysis with automatic version detection and reporting
**Usage**:

```bash
sbatch scripts/slurm_python.sh
```

**Output**: `logs/cluster_performance_report.txt`

## Quick Start Commands

1. **Transfer files to cluster**:

   ```bash
   scp -r K-means/ username@login.hpc.di.uniroma1.it:~/
   ```

2. **Submit a basic performance test**:

   ```bash
   cd K-means
   sbatch scripts/slurm_omp_performance.sh
   ```

3. **Check job status**:

   ```bash
   squeue -u $USER
   ```

4. **View results**:
   ```bash
   cat logs/slurm_performance_results.txt
   ```

## Script Customization

### Memory Requirements

Adjust memory based on dataset size:

- Small datasets (2D, 10D): `--mem=4GB`
- Medium datasets (20D): `--mem=8GB`
- Large datasets (100D+): `--mem=16GB`

### Time Limits

Typical runtime estimates:

- Basic OpenMP test: 30-60 minutes
- Comprehensive analysis: 1-2 hours
- CUDA test: 30-60 minutes

### CPU Count

Maximum recommended for students partition:

```bash
#SBATCH --cpus-per-task=8
```

## Output Files

### Performance Logs

- `logs/slurm_performance_results.txt`: Detailed OpenMP scaling results
- `logs/slurm_comprehensive_results.txt`: Multi-version comparison
- `logs/slurm_cuda_results.txt`: GPU vs CPU performance
- `logs/cluster_performance_report.txt`: Python tool comprehensive report

### SLURM System Logs

- `logs/slurm_*_<job_id>.out`: Standard output
- `logs/slurm_*_<job_id>.err`: Error output

### Result Files

- `results/result_*_slurm.out`: K-means clustering results
- Individual result files for each configuration tested

## Troubleshooting

### Common Issues

1. **Build Failures**:

   - Ensure `gcc` with OpenMP support: `gcc -fopenmp`
   - For MPI: `module load mpi` or use `mpicc`
   - For CUDA: `module load cuda` or use `nvcc`

2. **Memory Errors**:

   - Increase `--mem` parameter
   - Check dataset size vs available memory

3. **Time Limit Exceeded**:

   - Reduce number of test iterations
   - Use smaller datasets
   - Increase `--time` parameter

4. **GPU Not Available**:
   - Ensure `--gres=gpu:1` in SLURM script
   - Check GPU partition availability

### Debug Mode

Add debugging to any script:

```bash
#SBATCH --verbose
set -x  # Add to script for detailed execution trace
```

## Performance Optimization Tips

1. **Thread Binding**: Scripts use `OMP_PROC_BIND=true` and `OMP_PLACES=cores`
2. **Optimal Thread Count**: Usually 4-8 threads for CPU-bound tasks
3. **Dataset Selection**: Start with smaller datasets for initial testing
4. **Multiple Runs**: Scripts perform multiple runs and take best times

## Example Workflow

```bash
# 1. Transfer and setup
scp -r K-means/ user@cluster:~/
ssh user@cluster
cd K-means

# 2. Quick OpenMP test
sbatch scripts/slurm_omp_performance.sh

# 3. Monitor job
watch -n 10 squeue -u $USER

# 4. View results when complete
cat logs/slurm_performance_results.txt

# 5. Comprehensive analysis
sbatch scripts/slurm_python.sh

# 6. Final report
cat logs/cluster_performance_report.txt
```

## Resource Usage Guidelines

- **Students Partition**: Use for development and testing
- **Department Partition**: Use for production runs (if available)
- **GPU Resources**: Reserve for CUDA-specific tests only
- **Parallel Jobs**: Submit multiple small jobs rather than one large job when possible
