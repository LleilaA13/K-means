#!/bin/bash
#SBATCH --job-name=kmeans_mpi_omp_performance
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_mpi_omp_%j.out
#SBATCH --error=logs/slurm_mpi_omp_%j.err

# K-means MPI+OpenMP Hybrid Performance Analysis for Sapienza HPC Cluster
# This script tests MPI+OpenMP hybrid scaling on a single node following cluster guidelines:
# - Multicore partition: single node only, max 64 cores total, 6 hour limit

# ============================================
# CONFIGURATION: Change this to test different datasets
# ============================================
INPUT_DATASET="data/200k_100.inp"
# Alternative options:
# INPUT_DATASET="data/input2D.inp"
# INPUT_DATASET="data/input10D.inp"
# INPUT_DATASET="data/input20D.inp"
# INPUT_DATASET="data/input100D.inp"

# ============================================
# MPI+OpenMP HYBRID CONFIGURATION: Single node with 64 cores
# ============================================
# Test configurations: "processes:threads" format (theoretically reasonable only)
HYBRID_CONFIGS="1:1 1:4 1:8 1:16 1:32 1:64 2:8 2:16 2:32 4:4 4:8 4:16"

# ============================================
# SLURM CONFIGURATION: Single node multicore partition
# ============================================
# NOTE: To change SLURM directives, you need to edit the #SBATCH lines at the top
# Current settings follow multicore partition constraints:
#   Nodes: 1 (--nodes=1) - Multicore partition allows only single node
#   Cores per task: 64 (--cpus-per-task=64) - Full node capacity
#   Time: 6 hours (--time=06:00:00) - Multicore partition limit
#   Partition: multicore (--partition=multicore)
#   Total cores available: 1 × 64 = 64

echo "=== K-means MPI+OpenMP Performance Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Dataset: $INPUT_DATASET"
echo "Configurations: $(echo $HYBRID_CONFIGS | wc -w) to test"
echo "Start: $(date)"
echo "============================================"

# Create necessary directories
mkdir -p logs results

# Clear previous logs
> logs/timing_log_mpi_omp.txt
> logs/slurm_mpi_omp_performance_results.txt

# Log system information
echo "=== System Information ===" >> logs/slurm_mpi_omp_performance_results.txt
echo "Job ID: $SLURM_JOB_ID" >> logs/slurm_mpi_omp_performance_results.txt
echo "Nodes: $SLURM_NODELIST" >> logs/slurm_mpi_omp_performance_results.txt
echo "Date: $(date)" >> logs/slurm_mpi_omp_performance_results.txt
echo "CPU Info:" >> logs/slurm_mpi_omp_performance_results.txt
lscpu >> logs/slurm_mpi_omp_performance_results.txt
echo "" >> logs/slurm_mpi_omp_performance_results.txt

# Build the MPI+OpenMP application if not already built
if [ ! -f "build/KMEANS_mpi_omp" ]; then
    echo "Building K-means MPI+OpenMP version..."
    mkdir -p build
    mpicc -O3 -Wall -fopenmp src/KMEANS_mpi_omp.c -lm -o build/KMEANS_mpi_omp
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build MPI+OpenMP version"
        exit 1
    fi
fi

# Also build sequential version for comparison
if [ ! -f "build/KMEANS_seq" ]; then
    echo "Building K-means sequential version..."
    gcc -O3 -Wall src/KMEANS.c -lm -o build/KMEANS_seq
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build sequential version"
        exit 1
    fi
fi

# Check if input file exists
if [ ! -f "$INPUT_DATASET" ]; then
    echo "ERROR: Input file $INPUT_DATASET not found"
    echo "Available datasets:"
    ls -la data/*.inp 2>/dev/null || echo "  No .inp files found in data/ directory"
    exit 1
fi

# Extract dataset name for output files
DATASET_NAME=$(basename "$INPUT_DATASET" .inp)

echo "=== Running Sequential Baseline ===" | tee -a logs/slurm_mpi_omp_performance_results.txt
echo "Dataset: $INPUT_DATASET" | tee -a logs/slurm_mpi_omp_performance_results.txt

# Run sequential version for baseline
echo "Running sequential baseline..." | tee -a logs/slurm_mpi_omp_performance_results.txt
seq_output_file="results/result_${DATASET_NAME}_seq_mpi_omp_slurm.out"
seq_timing_file="${seq_output_file}.timing"
./build/KMEANS_seq "$INPUT_DATASET" 20 100 1.0 0.0001 "$seq_output_file" 42
# Read computation time from timing file
if [ -f "$seq_timing_file" ]; then
    seq_time=$(grep "computation_time:" "$seq_timing_file" | cut -d' ' -f2)
else
    echo "ERROR: Timing file $seq_timing_file not found"
    exit 1
fi
echo "Sequential computation time: $seq_time seconds" | tee -a logs/slurm_mpi_omp_performance_results.txt

echo "" | tee -a logs/slurm_mpi_omp_performance_results.txt
echo "=== Running MPI+OpenMP Hybrid Performance Tests ===" | tee -a logs/slurm_mpi_omp_performance_results.txt
printf "%-12s %-8s %-10s %-12s %-8s %-12s\n" "Config(P:T)" "Total" "Processes" "Time(s)" "Speedup" "Efficiency(%)" | tee -a logs/slurm_mpi_omp_performance_results.txt
echo "-------------------------------------------------------------------------" | tee -a logs/slurm_mpi_omp_performance_results.txt

# Test different hybrid configurations
for config in $HYBRID_CONFIGS; do
    # Parse configuration
    processes=$(echo $config | cut -d':' -f1)
    threads=$(echo $config | cut -d':' -f2)
    total_threads=$((processes * threads))
    
    # Skip invalid configurations
    if [ $total_threads -gt $SLURM_CPUS_PER_TASK ]; then
        continue
    fi
    
    echo "Testing $config ($total_threads threads)..."
    
    # Set OpenMP environment variables
    export OMP_NUM_THREADS=$threads
    export OMP_PROC_BIND=true
    export OMP_PLACES=cores
    export OMP_DYNAMIC=false
    export OMP_NESTED=false
    export OMP_SCHEDULE=dynamic,64
    export OMP_WAIT_POLICY=active
    
    # Run 3 times, keep best time
    best_time=999999
    for run in 1 2 3; do
        
        # Define output and timing files
        output_file="results/result_${DATASET_NAME}_mpi_omp_${processes}p_${threads}t_run${run}_slurm.out"
        timing_file="${output_file}.timing"
        
        # Run the test on single node (multicore partition)
        srun --partition=multicore --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
             mpirun -np $processes --oversubscribe \
             ./build/KMEANS_mpi_omp "$INPUT_DATASET" 20 100 1.0 0.0001 \
             "$output_file" 42 $threads
        
        # Read timing
        if [ -f "$timing_file" ]; then
            current_time=$(grep "computation_time:" "$timing_file" | cut -d' ' -f2)
        else
            continue
        fi
        
        # Keep best time
        if (( $(echo "$current_time < $best_time" | bc -l) )); then
            best_time=$current_time
        fi
    done
    
    # Calculate speedup and efficiency
    speedup=$(echo "scale=3; $seq_time / $best_time" | bc -l)
    efficiency=$(echo "scale=1; $speedup * 100 / $total_threads" | bc -l)
    
    # Log results
    printf "%-12s %-8s %-10s %-12.3f %-8.3f %-12.1f\n" "$config" "$total_threads" "$processes" "$best_time" "$speedup" "$efficiency" | tee -a logs/slurm_mpi_omp_performance_results.txt
done

echo "" | tee -a logs/slurm_mpi_omp_performance_results.txt
echo "=== Performance Summary ===" | tee -a logs/slurm_mpi_omp_performance_results.txt

# Find best configuration
optimal_line=$(tail -n +4 logs/slurm_mpi_omp_performance_results.txt | grep -E "^[0-9]" | sort -k5 -nr | head -n 1)
if [ -n "$optimal_line" ]; then
    optimal_config=$(echo "$optimal_line" | awk '{print $1}')
    optimal_speedup=$(echo "$optimal_line" | awk '{print $5}')
    echo "Best: $optimal_config with ${optimal_speedup}x speedup" | tee -a logs/slurm_mpi_omp_performance_results.txt
fi
echo "Sequential baseline: $seq_time seconds" | tee -a logs/slurm_mpi_omp_performance_results.txt

echo "Job completed: $(date)" | tee -a logs/slurm_mpi_omp_performance_results.txt

echo "============================================"
echo "Performance test complete!"
echo "Results: logs/slurm_mpi_omp_performance_results.txt"
