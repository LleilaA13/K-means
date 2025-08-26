#!/bin/bash
#SBATCH --job-name=kmeans_mpi_performance
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_mpi_%j.out
#SBATCH --error=logs/slurm_mpi_%j.err

# K-means MPI Performance Analysis for Sapienza HPC Cluster
# This script tests MPI scaling on a single node following cluster guidelines:
# - Multicore partition: single node only, max 64 cores per node, 6 hour limit

# ============================================
# CONFIGURATION: Change this to test different datasets
# ============================================
INPUT_DATASET="data/input100D2.inp"
# Alternative options:
# INPUT_DATASET="data/input2D.inp"
# INPUT_DATASET="data/input10D.inp"
# INPUT_DATASET="data/input20D.inp"
# INPUT_DATASET="data/input100D.inp"

# ============================================
# MPI CONFIGURATION: Single node with 64 cores
# ============================================
MAX_PROCESSES=64                        # Max processes on single node (64 cores)
PROCESS_COUNTS="1 2 4 8 16 32 64"      # Space-separated list of process counts to test
# Multicore partition: single node, max 64 cores available

# Alternative examples:
# PROCESS_COUNTS="1 4 8 16 32"          # Powers of 2 only
# PROCESS_COUNTS="8 16 24 32 48 64"     # Higher core counts only

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

echo "=== K-means MPI Performance Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Cores per task: $SLURM_CPUS_PER_TASK"
echo "Total cores available: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Dataset: $INPUT_DATASET"
echo "Max processes configured: $MAX_PROCESSES"
echo "Process counts to test: $PROCESS_COUNTS"
echo "Start time: $(date)"
echo "============================================"

# Validate configuration
if [ "$MAX_PROCESSES" -gt "$SLURM_CPUS_PER_TASK" ]; then
    echo "WARNING: MAX_PROCESSES ($MAX_PROCESSES) exceeds available cores ($SLURM_CPUS_PER_TASK)"
    echo "Consider updating MAX_PROCESSES or the SLURM configuration"
fi

# Create necessary directories
mkdir -p logs results

# Clear previous logs
> logs/timing_log_mpi.txt
> logs/slurm_mpi_performance_results.txt

# Log system information
echo "=== System Information ===" >> logs/slurm_mpi_performance_results.txt
echo "Job ID: $SLURM_JOB_ID" >> logs/slurm_mpi_performance_results.txt
echo "Nodes: $SLURM_NODELIST" >> logs/slurm_mpi_performance_results.txt
echo "Date: $(date)" >> logs/slurm_mpi_performance_results.txt
echo "CPU Info:" >> logs/slurm_mpi_performance_results.txt
lscpu >> logs/slurm_mpi_performance_results.txt
echo "" >> logs/slurm_mpi_performance_results.txt

# Build the MPI application if not already built
if [ ! -f "build/KMEANS_mpi" ]; then
    echo "Building K-means MPI version..."
    mkdir -p build
    mpicc -O3 -Wall src/KMEANS_mpi.c -lm -o build/KMEANS_mpi
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build MPI version"
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

echo "=== Running Sequential Baseline ===" | tee -a logs/slurm_mpi_performance_results.txt
echo "Dataset: $INPUT_DATASET" | tee -a logs/slurm_mpi_performance_results.txt

# Run sequential version for baseline
echo "Running sequential baseline..." | tee -a logs/slurm_mpi_performance_results.txt
seq_output_file="results/result_${DATASET_NAME}_seq_mpi_slurm.out"
seq_timing_file="${seq_output_file}.timing"
./build/KMEANS_seq "$INPUT_DATASET" 20 100 1.0 0.0001 "$seq_output_file" 42
# Read computation time from timing file
if [ -f "$seq_timing_file" ]; then
    seq_time=$(grep "computation_time:" "$seq_timing_file" | cut -d' ' -f2)
else
    echo "ERROR: Timing file $seq_timing_file not found"
    exit 1
fi
echo "Sequential computation time: $seq_time seconds" | tee -a logs/slurm_mpi_performance_results.txt

echo "" | tee -a logs/slurm_mpi_performance_results.txt
echo "=== Running MPI Performance Tests ===" | tee -a logs/slurm_mpi_performance_results.txt
printf "%-10s %-12s %-8s %-12s\n" "Processes" "Time(s)" "Speedup" "Efficiency(%)" | tee -a logs/slurm_mpi_performance_results.txt
echo "----------------------------------------------------" | tee -a logs/slurm_mpi_performance_results.txt

# Test different process counts (configurable)
for processes in $PROCESS_COUNTS; do
    # Validate process count doesn't exceed available cores
    if [ $processes -gt $MAX_PROCESSES ]; then
        echo "Warning: Skipping $processes processes (exceeds MAX_PROCESSES=$MAX_PROCESSES)"
        continue
    fi
    
    echo "Testing with $processes MPI processes..."
    
    # Run the test 3 times and take the best time
    best_time=999999
    for run in 1 2 3; do
        echo "  Run $run with $processes processes..."
        
        # Define output and timing files
        output_file="results/result_${DATASET_NAME}_mpi_${processes}p_run${run}_slurm.out"
        timing_file="${output_file}.timing"
        
        # Run the test on single node (multicore partition)
        srun --partition=multicore --nodes=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
             mpirun -np $processes --oversubscribe \
             ./build/KMEANS_mpi "$INPUT_DATASET" 20 100 1.0 0.0001 \
             "$output_file" 42
        
        # Read computation time from timing file
        if [ -f "$timing_file" ]; then
            current_time=$(grep "computation_time:" "$timing_file" | cut -d' ' -f2)
        else
            echo "    ERROR: Timing file $timing_file not found"
            continue
        fi
        
        # Keep the best (minimum) time
        if (( $(echo "$current_time < $best_time" | bc -l) )); then
            best_time=$current_time
        fi
        
        echo "    Computation time: $current_time seconds"
    done
    
    # Calculate speedup and efficiency
    speedup=$(echo "scale=3; $seq_time / $best_time" | bc -l)
    efficiency=$(echo "scale=1; $speedup * 100 / $processes" | bc -l)
    
    # Log results
    printf "%-10s %-12.3f %-8.3f %-12.1f\n" "$processes" "$best_time" "$speedup" "$efficiency" | tee -a logs/slurm_mpi_performance_results.txt
    echo "Best computation time with $processes processes: $best_time seconds (speedup: ${speedup}x, efficiency: ${efficiency}%)"
done

echo "" | tee -a logs/slurm_mpi_performance_results.txt
echo "=== Performance Summary ===" | tee -a logs/slurm_mpi_performance_results.txt

# Find optimal configuration
optimal_line=$(tail -n +4 logs/slurm_mpi_performance_results.txt | grep -E "^[0-9]" | sort -k3 -nr | head -n 1)
if [ -n "$optimal_line" ]; then
    optimal_processes=$(echo "$optimal_line" | awk '{print $1}')
    optimal_speedup=$(echo "$optimal_line" | awk '{print $3}')
    echo "Best configuration: $optimal_processes processes with ${optimal_speedup}x speedup" | tee -a logs/slurm_mpi_performance_results.txt
fi
echo "Sequential baseline (computation only): $seq_time seconds" | tee -a logs/slurm_mpi_performance_results.txt

echo "" | tee -a logs/slurm_mpi_performance_results.txt
echo "Job completed at: $(date)" | tee -a logs/slurm_mpi_performance_results.txt

echo "============================================"
echo "Performance analysis complete!"
echo "Dataset tested: $INPUT_DATASET"
echo "Results saved in: logs/slurm_mpi_performance_results.txt"
echo "SLURM output: logs/slurm_mpi_${SLURM_JOB_ID}.out"
echo "Individual results: results/result_${DATASET_NAME}_*_slurm.out"
echo "Timing files: results/result_${DATASET_NAME}_*.timing"
echo "NOTE: Performance statistics based on algorithm computation time from .timing files"
