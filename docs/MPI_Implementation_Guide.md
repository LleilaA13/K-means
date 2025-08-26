# K-means MPI Implementation Guide

## Overview

This document provides a comprehensive guide for running K-means clustering using MPI (Message Passing Interface) and MPI+OpenMP hybrid parallelization on the Sapienza HPC cluster. Both implementations follow the cluster guidelines from the [Sapienza HPC User Guide](https://di-uniroma1-hpc-user-guide.netlify.app/user_guide/running_multicore_apps).

## Implementation Variants

### 1. Pure MPI Version (`KMEANS_mpi.c`)

- **Parallelization Strategy**: Inter-process communication across nodes
- **Data Distribution**: Each MPI process handles a subset of data points
- **Communication**: MPI_Allreduce for centroid updates and change counting
- **Best Use Case**: Multi-node scaling, distributed memory systems

### 2. MPI+OpenMP Hybrid Version (`KMEANS_mpi_omp.c`)

- **Parallelization Strategy**: MPI for inter-node + OpenMP for intra-node
- **Two-Level Parallelism**:
  - MPI processes communicate between nodes
  - OpenMP threads parallelize within each process
- **Best Use Case**: Optimal utilization of multi-core nodes in clusters

## Key Technical Features

### MPI Implementation Details

#### Data Distribution Strategy

```c
// Calculate local data size per process
int local_lines = lines / size;
int remainder = lines % size;

// First 'remainder' processes get one extra row
if (rank < remainder) {
    local_lines++;
}
```

#### Communication Patterns

- **Non-blocking Allreduce**: `MPI_Iallreduce` for change counting
- **Blocking Allreduce**: `MPI_Allreduce` for centroid accumulation
- **Gatherv**: `MPI_Gatherv` for final result collection

### Hybrid (MPI+OpenMP) Enhancements

#### OpenMP Integration Points

```c
// Parallel distance calculation
#pragma omp parallel for private(j, class, minDist, dist) reduction(+ : local_changes)
for (i = 0; i < local_lines; i++) {
    // Distance computation and assignment
}

// Parallel local centroid accumulation
#pragma omp parallel for private(j, class)
for (i = 0; i < local_lines; i++) {
    // Local centroid updates with atomic operations
}
```

#### Memory Management

- Thread-safe atomic operations for shared data structures
- Proper synchronization between MPI and OpenMP layers

## Performance Analysis Scripts

### 1. MPI Performance Testing (`slurm_mpi_performance.sh`)

#### Test Configurations

| Nodes | Processes/Node | Total Processes | Description         |
| ----- | -------------- | --------------- | ------------------- |
| 1     | 1,2,4,8        | 1,2,4,8         | Single-node scaling |
| 2     | 1,2,4,8        | 2,4,8,16        | Multi-node scaling  |

#### Key Features

- **Sequential Baseline**: Establishes performance reference
- **Multiple Runs**: 3 runs per configuration, best time reported
- **Comprehensive Metrics**: Speedup and efficiency calculations
- **Result Logging**: Detailed performance analysis

### 2. Hybrid Performance Testing (`slurm_mpi_omp_performance.sh`)

#### Hybrid Configurations

| Nodes | MPI Procs/Node | OMP Threads/Proc | Total Threads | Strategy                              |
| ----- | -------------- | ---------------- | ------------- | ------------------------------------- |
| 1     | 1              | 1,2,4,8          | 1,2,4,8       | Pure OpenMP within single MPI process |
| 1     | 2,4            | 1,2,4            | 2,4,8         | Hybrid within single node             |
| 2     | 1,2,4          | 1,2,4            | 2,4,8,16      | Multi-node hybrid                     |

#### Advanced Features

- **Thread Affinity**: Optimal core binding (`OMP_PROC_BIND=spread`)
- **NUMA Awareness**: Considers node architecture
- **Configuration Validation**: Ensures no oversubscription
- **Comparative Analysis**: Pure MPI vs Hybrid performance

## Cluster Usage Guidelines

### Building Applications

#### Sequential and MPI Versions

```bash
# Build all versions
make clean-build
make KMEANS_seq KMEANS_mpi KMEANS_mpi_omp

# Verify builds
./scripts/utils/test_mpi_build.sh
```

#### Compilation Commands (on cluster)

```bash
# MPI version
mpicc -O3 -Wall src/KMEANS_mpi.c -lm -o build/KMEANS_mpi

# MPI+OpenMP hybrid version
mpicc -O3 -Wall -fopenmp src/KMEANS_mpi_omp.c -lm -o build/KMEANS_mpi_omp
```

### Running Applications

#### SLURM Job Submission

```bash
# MPI performance analysis
./scripts/run.sh slurm mpi_perf

# MPI+OpenMP hybrid analysis
./scripts/run.sh slurm mpi_omp_perf

# Check job status
squeue -u $USER
```

#### Manual Execution Examples

```bash
# Pure MPI (following Sapienza guidelines)
srun --partition=students --nodes=2 --cpus-per-task=4 \
     mpirun -np 8 --oversubscribe \
     ./build/KMEANS_mpi data/input100D2.inp 4 100 1 0.001 results/output_mpi.out

# MPI+OpenMP hybrid
export OMP_NUM_THREADS=2
srun --partition=students --nodes=2 --cpus-per-task=4 \
     mpirun -np 4 --oversubscribe \
     ./build/KMEANS_mpi_omp data/input100D2.inp 4 100 1 0.001 results/output_hybrid.out
```

## Performance Optimization Strategies

### 1. MPI Optimization

- **Load Balancing**: Even distribution of data points across processes
- **Communication Minimization**: Non-blocking operations where possible
- **Memory Efficiency**: Local data storage reduces memory footprint per process

### 2. Hybrid Optimization

- **Thread Placement**: `OMP_PROC_BIND=spread` for optimal core utilization
- **Memory Access**: Thread-local data structures reduce contention
- **Synchronization**: Minimal critical sections in OpenMP regions

### 3. Cluster-Specific Tuning

- **Node Architecture**: 8 cores per node in students partition
- **Memory Hierarchy**: Consider NUMA effects on 2+ node jobs
- **Network Topology**: Optimize for inter-node communication patterns

## Expected Performance Characteristics

### Pure MPI Scaling

- **Strong Scaling**: Effective up to 8-16 processes
- **Multi-node Efficiency**: Good performance across 2 nodes
- **Communication Overhead**: Increases with process count

### Hybrid Scaling Advantages

- **Memory Efficiency**: Fewer MPI processes reduce memory overhead
- **Cache Performance**: OpenMP threads share cache within nodes
- **Network Reduction**: Less inter-node communication

### Optimal Configurations

Based on cluster architecture (8 cores per node):

#### Single Node (students partition)

- **Pure MPI**: 4-8 processes
- **Hybrid**: 2 MPI × 4 OpenMP or 4 MPI × 2 OpenMP

#### Multi-Node

- **Pure MPI**: 4 processes per node (8 total)
- **Hybrid**: 2 MPI × 2 OpenMP per node (8 processes, 16 threads total)

## Monitoring and Analysis

### Performance Metrics

- **Speedup**: Sequential time / Parallel time
- **Efficiency**: (Speedup / Total processes) × 100%
- **Scalability**: Performance retention as processes increase

### Log Analysis

```bash
# View MPI results
cat logs/slurm_mpi_performance_results.txt

# View hybrid results
cat logs/slurm_mpi_omp_performance_results.txt

# Check SLURM job output
cat logs/slurm_mpi_*.out
```

### Result Verification

```bash
# Compare MPI results against sequential
python3 scripts/analysis/compare_module.py \
    results/result_seq.out results/result_mpi.out

# Verify hybrid correctness
python3 scripts/analysis/verify_results.py
```

## Troubleshooting

### Common Issues

#### Build Problems

```bash
# If MPI compiler not found
module load mpi/openmpi  # (if using modules)
which mpicc               # Verify MPI compiler available
```

#### Runtime Issues

```bash
# Check MPI processes
ps aux | grep mpi

# Verify OpenMP threads (in hybrid mode)
export OMP_DISPLAY_ENV=true
```

#### Performance Issues

- **Memory**: Check if processes fit in node memory
- **Oversubscription**: Ensure total threads ≤ available cores
- **Load Balance**: Verify even data distribution

## Integration with Existing Workflow

### Scripts Integration

The MPI scripts integrate seamlessly with the existing K-means analysis workflow:

```bash
# Complete performance analysis pipeline
./scripts/run.sh slurm omp_perf      # OpenMP analysis
./scripts/run.sh slurm mpi_perf      # MPI analysis
./scripts/run.sh slurm mpi_omp_perf  # Hybrid analysis

# Compare all implementations
python3 scripts/analysis/verify_results.py
```

### Result Compatibility

- All implementations produce identical clustering results
- Performance logs follow consistent format
- Results integrate with existing visualization tools

---

_Document created: August 25, 2025_  
_Version: 1.0_  
_Compatibility: Sapienza HPC Cluster (students partition)_
