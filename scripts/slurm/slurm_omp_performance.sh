#!/bin/bash
#SBATCH --job-name=kmeans_omp_performance
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_omp_%j.out
#SBATCH --error=logs/slurm_omp_%j.err

# K-means OpenMP Performance Analysis for Sapienza HPC Cluster
# This script tests OpenMP scaling following multicore partition guidelines

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
# THREAD CONFIGURATION: Following multicore partition guidelines
# ============================================
MAX_THREADS=64                          # Must match --cpus-per-task above
THREAD_COUNTS="1 2 4 8 16 32 64"       # Space-separated list of thread counts to test
# Multicore partition: max 64 threads per process
# Alternative examples:
# THREAD_COUNTS="1 4 8 16 32 64"    # Powers of 2 only
# THREAD_COUNTS="8 16 24 32 48 64"  # High thread counts only
# THREAD_COUNTS="1 8 16 32 64"      # Fewer test points

# ============================================
# SLURM CONFIGURATION: Following multicore partition guidelines
# ============================================
# NOTE: To change SLURM directives, you need to edit the #SBATCH lines at the top
# Current settings follow multicore partition guidelines:
#   Cores: 64 (--cpus-per-task=64) - Full node capacity in multicore
#   Time: 6 hours (--time=06:00:00) - Multicore partition limit
#   Partition: multicore (--partition=multicore)

echo "=== K-means OpenMP Performance Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores available: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Dataset: $INPUT_DATASET"
echo "Max threads configured: $MAX_THREADS"
echo "Thread counts to test: $THREAD_COUNTS"
echo "Start time: $(date)"
echo "============================================"

# Validate configuration
if [ "$MAX_THREADS" -ne "$SLURM_CPUS_PER_TASK" ]; then
    echo "WARNING: MAX_THREADS ($MAX_THREADS) doesn't match SLURM --cpus-per-task ($SLURM_CPUS_PER_TASK)"
    echo "Consider updating MAX_THREADS or the #SBATCH --cpus-per-task directive"
fi

# Create necessary directories
mkdir -p logs results

# Clear previous logs
> logs/timing_log_omp.txt
> logs/slurm_performance_results.txt

# Log system information
echo "=== System Information ===" >> logs/slurm_performance_results.txt
echo "Node: $SLURM_NODELIST" >> logs/slurm_performance_results.txt
echo "Date: $(date)" >> logs/slurm_performance_results.txt
echo "CPU Info:" >> logs/slurm_performance_results.txt
lscpu >> logs/slurm_performance_results.txt
echo "" >> logs/slurm_performance_results.txt

# Build the application if not already built
if [ ! -f "build/KMEANS_omp" ]; then
    echo "Building K-means OpenMP version..."
    mkdir -p build
    gcc -Wall -fopenmp src/KMEANS_omp.c -lm -o build/KMEANS_omp
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build OpenMP version"
        exit 1
    fi
fi

# Also build sequential version for comparison
if [ ! -f "build/KMEANS_seq" ]; then
    echo "Building K-means sequential version..."
    gcc -Wall src/KMEANS.c -lm -o build/KMEANS_seq
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

echo "=== Running Sequential Baseline ===" | tee -a logs/slurm_performance_results.txt
echo "Dataset: $INPUT_DATASET" | tee -a logs/slurm_performance_results.txt

# Run sequential version for baseline
echo "Running sequential baseline..." | tee -a logs/slurm_performance_results.txt
seq_output_file="results/result_${DATASET_NAME}_seq_slurm.out"
seq_timing_file="${seq_output_file}.timing"
./build/KMEANS_seq "$INPUT_DATASET" 20 100 1.0 0.0001 "$seq_output_file" 42
# Read computation time from timing file
if [ -f "$seq_timing_file" ]; then
    seq_time=$(grep "computation_time:" "$seq_timing_file" | cut -d' ' -f2)
else
    echo "ERROR: Timing file $seq_timing_file not found"
    exit 1
fi
echo "Sequential computation time: $seq_time seconds" | tee -a logs/slurm_performance_results.txt

echo "" | tee -a logs/slurm_performance_results.txt
echo "=== Running OpenMP Performance Tests ===" | tee -a logs/slurm_performance_results.txt
printf "%-8s %-12s %-8s %-12s\n" "Threads" "Time(s)" "Speedup" "Efficiency(%)" | tee -a logs/slurm_performance_results.txt
echo "----------------------------------------------------" | tee -a logs/slurm_performance_results.txt

# Test different thread counts (configurable)
for threads in $THREAD_COUNTS; do
    # Validate thread count doesn't exceed available cores
    if [ $threads -gt $MAX_THREADS ]; then
        echo "Warning: Skipping $threads threads (exceeds MAX_THREADS=$MAX_THREADS)"
        continue
    fi
    
    echo "Testing with $threads threads..."
    
    # Optimized OpenMP environment variables for better performance
    export OMP_NUM_THREADS=$threads
    export OMP_PROC_BIND=spread        # Spread threads across NUMA domains
    export OMP_PLACES=numa_domains     # Use NUMA-aware placement
    export OMP_DYNAMIC=false          # Disable dynamic thread adjustment
    export OMP_NESTED=false           # Disable nested parallelism
    export OMP_SCHEDULE=dynamic,64     # Dynamic scheduling works better for uneven workloads like K-means
    export OMP_WAIT_POLICY=active     # Keep threads active (better for short parallel regions)
    
    # Run the test 3 times and take the best time
    best_time=999999
    for run in 1 2 3; do
        echo "  Run $run with $threads threads..."
        
        # Define output and timing files
        output_file="results/result_${DATASET_NAME}_omp_${threads}t_run${run}_slurm.out"
        timing_file="${output_file}.timing"
        
        # Run the test
        ./build/KMEANS_omp "$INPUT_DATASET" 20 100 1.0 0.0001 "$output_file" 42 $threads
        
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
    efficiency=$(echo "scale=1; $speedup * 100 / $threads" | bc -l)
    
    # Log results
    printf "%-8s %-12.3f %-8.3f %-12.1f\n" "$threads" "$best_time" "$speedup" "$efficiency" | tee -a logs/slurm_performance_results.txt
    echo "Best computation time with $threads threads: $best_time seconds (speedup: ${speedup}x, efficiency: ${efficiency}%)"
done

echo "" | tee -a logs/slurm_performance_results.txt
echo "=== Performance Summary ===" | tee -a logs/slurm_performance_results.txt

# Find optimal configuration
optimal_line=$(tail -n +4 logs/slurm_performance_results.txt | head -n 4 | sort -k3 -nr | head -n 1)
optimal_threads=$(echo "$optimal_line" | cut -f1)
optimal_speedup=$(echo "$optimal_line" | cut -f3)

echo "Best configuration: $optimal_threads threads with ${optimal_speedup}x speedup" | tee -a logs/slurm_performance_results.txt
echo "Sequential baseline (computation only): $seq_time seconds" | tee -a logs/slurm_performance_results.txt

echo "" | tee -a logs/slurm_performance_results.txt
echo "Job completed at: $(date)" | tee -a logs/slurm_performance_results.txt

echo "============================================"
echo "Performance analysis complete!"
echo "Dataset tested: $INPUT_DATASET"
echo "Results saved in: logs/slurm_performance_results.txt"
echo "SLURM output: logs/slurm_omp_${SLURM_JOB_ID}.out"
echo "Individual results: results/result_${DATASET_NAME}_*_slurm.out"
echo "Timing files: results/result_${DATASET_NAME}_*.timing"
echo "NOTE: Performance statistics based on algorithm computation time from .timing files"
