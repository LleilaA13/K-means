# SLURM Scripts for Sapienza University HPC Cluster

This directory contains optimized SLURM job scripts for running K-means performance analysis on the Sapienza University HPC cluster.

## Cluster Specifications

### Students Partition (Recommended)

- **Walltime**: 24 hours maximum
- **CPUs**: 8 cores maximum per job
- **Memory**: 8-32GB available
- **Queue Priority**: Standard

### Department Only Partition (If Available)

- **Walltime**: 72 hours maximum
- **CPUs**: Unlimited cores
- **Memory**: Higher limits
- **Queue Priority**: Higher

### GPU Resources

- **Hardware**: RTX Quadro 6000
- **Access**: Request with `--gres=gpu:1`
- **CUDA**: sm_75 architecture

## Available Scripts

### 🚀 Quick Start: `slurm_omp_performance.sh`

**Purpose**: Fast OpenMP scaling test (recommended first run)

```bash
sbatch slurm_omp_performance.sh
```

**What it does**:

- Tests 1, 2, 4, 8 threads
- Runs multiple iterations for accuracy
- Generates performance summary
- ~30-60 minutes runtime

**Output**: `../logs/slurm_performance_results.txt`

### 🔬 Comprehensive: `slurm_comprehensive.sh`

**Purpose**: Complete multi-version analysis

```bash
sbatch slurm_comprehensive.sh
```

**What it does**:

- Tests Sequential, OpenMP, MPI versions
- Multiple datasets (2D, 10D, 20D, 100D)
- Cross-platform performance comparison
- ~1-2 hours runtime

**Output**: `../logs/slurm_comprehensive_results.txt`

### 🎮 GPU Testing: `slurm_cuda.sh`

**Purpose**: CUDA vs CPU performance comparison

```bash
sbatch slurm_cuda.sh
```

**What it does**:

- Compares GPU vs best CPU implementation
- RTX Quadro 6000 optimization
- Memory transfer analysis
- ~30-60 minutes runtime

**Output**: `../logs/slurm_cuda_results.txt`

### 🐍 Python Analysis: `slurm_python.sh`

**Purpose**: Automated comprehensive analysis with professional reporting

```bash
sbatch slurm_python.sh
```

**What it does**:

- Runs `cluster_analysis.py` tool
- Auto-detects available compilers
- Generates detailed performance report
- Professional visualization ready
- ~1-2 hours runtime

**Output**: `../logs/cluster_performance_report.txt`

## Usage Workflow

### 1. Transfer Files to Cluster

```bash
# From local machine
scp -r K-means/ username@login.hpc.di.uniroma1.it:~/

# SSH to cluster
ssh username@login.hpc.di.uniroma1.it
cd K-means
```

### 2. Submit Jobs

```bash
# Quick test first
sbatch scripts/slurm/slurm_omp_performance.sh

# Check job status
squeue -u $USER

# When complete, run comprehensive analysis
sbatch scripts/slurm/slurm_python.sh
```

### 3. Monitor Progress

```bash
# Watch job queue
watch -n 30 squeue -u $USER

# Check current log output
tail -f logs/slurm_omp_*.out

# View real-time performance results
tail -f logs/slurm_performance_results.txt
```

### 4. Retrieve Results

```bash
# View final results
cat logs/cluster_performance_report.txt

# Copy results back to local machine
scp -r username@cluster:~/K-means/logs/ ./cluster_results/
```

## Script Customization

### Memory Requirements

Adjust based on your dataset size:

```bash
# Small datasets (2D, 10D)
#SBATCH --mem=4GB

# Medium datasets (20D, 100D)
#SBATCH --mem=8GB

# Large datasets (custom)
#SBATCH --mem=16GB
```

### CPU Allocation

```bash
# Standard configuration
#SBATCH --cpus-per-task=8

# For memory-intensive tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
```

### Time Limits

```bash
# Quick tests
#SBATCH --time=01:00:00

# Comprehensive analysis
#SBATCH --time=02:00:00

# Full research runs
#SBATCH --time=06:00:00
```

## Environment Setup

### Compiler Modules

The scripts automatically handle:

- `gcc` with OpenMP support (`-fopenmp`)
- `mpicc` for MPI builds (if available)
- `nvcc` for CUDA builds (if available)

### Python Environment

```bash
# Scripts automatically detect and use:
module load python/3.8  # or system python3
```

### OpenMP Configuration

All scripts optimize OpenMP settings:

```bash
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
```

## Output Files Explained

### Performance Results

- **`slurm_performance_results.txt`**: OpenMP scaling data
- **`slurm_comprehensive_results.txt`**: Multi-version comparison
- **`cluster_performance_report.txt`**: Professional analysis report

### SLURM System Logs

- **`slurm_*_<job_id>.out`**: Standard output from job
- **`slurm_*_<job_id>.err`**: Error messages (usually empty)

### Clustering Results

- **`result_*_slurm.out`**: Actual clustering outputs
- **`timing_log_*.txt`**: Detailed timing information

## Performance Expectations

### Typical Results (Sapienza HPC)

Based on cluster testing:

| Version    | Threads/Procs | Expected Speedup |
| ---------- | ------------- | ---------------- |
| Sequential | 1             | 1.0x (baseline)  |
| OpenMP     | 4             | 2.5-3.5x         |
| OpenMP     | 8             | 3.0-4.5x         |
| MPI        | 4             | 2.0-3.0x         |
| CUDA       | GPU           | 5.0-15.0x        |

_Note: Results vary by dataset size and cluster load_

## Troubleshooting

### Job Not Starting

```bash
# Check queue status
sinfo -p students

# View job details
scontrol show job <job_id>

# Check resource availability
squeue -p students
```

### Build Failures

```bash
# Check available modules
module avail

# Load required modules
module load gcc/9.3.0
module load cuda/11.0

# Verify compiler versions
gcc --version
nvcc --version
```

### Memory Issues

```bash
# Monitor memory usage during job
ssh <compute_node>
htop

# Increase memory allocation
#SBATCH --mem=16GB
```

### Time Limit Exceeded

```bash
# Check actual runtime needs
sacct -j <job_id> --format=JobID,Elapsed,Timelimit

# Adjust time limit
#SBATCH --time=04:00:00
```

## Advanced Usage

### Batch Job Submission

```bash
# Submit multiple configurations
for dataset in 2D 10D 20D 100D; do
    sbatch --job-name=kmeans_${dataset} slurm_omp_performance.sh
done
```

### Resource Optimization

```bash
# CPU-bound tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB

# Memory-bound tasks
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# GPU tasks
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
```

### Dependency Chains

```bash
# Submit dependent jobs
job1=$(sbatch --parsable slurm_omp_performance.sh)
job2=$(sbatch --dependency=afterok:$job1 slurm_comprehensive.sh)
```

## Contact and Support

For cluster-specific issues:

- **HPC Support**: Contact Sapienza IT support
- **Script Issues**: Check logs in `../logs/` directory
- **Performance Questions**: Review generated reports

## Best Practices

1. **Start Small**: Use `slurm_omp_performance.sh` first
2. **Monitor Resources**: Check `squeue` and logs regularly
3. **Save Results**: Copy important outputs to persistent storage
4. **Document Runs**: Keep notes on successful configurations
5. **Clean Up**: Remove old result files periodically
