#!/bin/bash
#SBATCH --job-name=kmeans_comprehensive_test
#SBATCH --partition=multicore
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=6:00:00
#SBATCH --output=logs/comprehensive_test_%j.out
#SBATCH --error=logs/comprehensive_test_%j.err

# Comprehensive K-means Performance Testing Script
# Tests all implementations (sequential, OpenMP, MPI, MPI+OpenMP) across all larger datasets
# Collects computation times from internal timing mechanisms only
# Author: Automated testing script
# Date: $(date)

echo "=== Comprehensive K-means Performance Testing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores available: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "============================================"

# ============================================
# CONFIGURATION
# ============================================

# Datasets to test (larger datasets as requested)
DATASETS=(
    "data/input100D2.inp"
    "data/200k_100.inp" 
    "data/400k_100.inp"
    "data/800k_100.inp"
)

# Number of test runs per configuration
NUM_RUNS=5

# Standard K-means parameters
K_CLUSTERS=20
MAX_ITERATIONS=3000
THRESHOLD=1.0
TOLERANCE=0.0001
SEED=42

# Process/thread configurations (Powers of 2 only)
MPI_PROCESSES=(1 2 4 8 16 32 64)
OMP_THREADS=(1 2 4 8 16 32 64)
MPI_OMP_CONFIGS=(
    "1:1" "1:2" "1:4" "1:8" "1:16" "1:32" "1:64"
    "2:1" "2:2" "2:4" "2:8" "2:16" "2:32"
    "4:1" "4:2" "4:4" "4:8" "4:16"
)

# Maximum hardware limit (64 cores)
MAX_CORES=64

# ============================================
# SETUP
# ============================================

# Create necessary directories
mkdir -p logs results logs/timing_logs

# Initialize timing log files (one per implementation)
echo "# Sequential K-means Timing Results" > logs/timing_logs/sequential_times.log
echo "# Format: dataset,run,computation_time_seconds" >> logs/timing_logs/sequential_times.log
echo "dataset,run,computation_time" >> logs/timing_logs/sequential_times.log

echo "# OpenMP K-means Timing Results" > logs/timing_logs/openmp_times.log  
echo "# Format: dataset,threads,run,computation_time_seconds" >> logs/timing_logs/openmp_times.log
echo "dataset,threads,run,computation_time" >> logs/timing_logs/openmp_times.log

echo "# MPI K-means Timing Results" > logs/timing_logs/mpi_times.log
echo "# Format: dataset,processes,run,computation_time_seconds" >> logs/timing_logs/mpi_times.log
echo "dataset,processes,run,computation_time" >> logs/timing_logs/mpi_times.log

echo "# MPI+OpenMP K-means Timing Results" > logs/timing_logs/mpi_openmp_times.log
echo "# Format: dataset,processes,threads,total_cores,run,computation_time_seconds" >> logs/timing_logs/mpi_openmp_times.log
echo "dataset,processes,threads,total_cores,run,computation_time" >> logs/timing_logs/mpi_openmp_times.log

# ============================================
# BUILD EXECUTABLES
# ============================================

echo "=== Building Executables ==="

# Sequential version
if [ ! -f "build/KMEANS_seq" ]; then
    echo "Building sequential version..."
    gcc -O3 -Wall src/KMEANS.c -lm -o build/KMEANS_seq
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build sequential version"
        exit 1
    fi
fi

# OpenMP version
if [ ! -f "build/KMEANS_omp" ]; then
    echo "Building OpenMP version..."
    gcc -O3 -Wall -fopenmp src/KMEANS_omp.c -lm -o build/KMEANS_omp
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build OpenMP version"
        exit 1
    fi
fi

# MPI version
if [ ! -f "build/KMEANS_mpi" ]; then
    echo "Building MPI version..."
    mpicc -O3 -Wall src/KMEANS_mpi.c -lm -o build/KMEANS_mpi
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build MPI version"
        exit 1
    fi
fi

# MPI+OpenMP version
if [ ! -f "build/KMEANS_mpi_omp" ]; then
    echo "Building MPI+OpenMP version..."
    mpicc -O3 -Wall -fopenmp src/KMEANS_mpi_omp.c -lm -o build/KMEANS_mpi_omp
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build MPI+OpenMP version"
        exit 1
    fi
fi

echo "All executables built successfully."
echo ""

# ============================================
# HELPER FUNCTIONS
# ============================================

# Function to extract computation time from timing file
extract_timing() {
    local timing_file="$1"
    if [ -f "$timing_file" ]; then
        grep "computation_time:" "$timing_file" | cut -d' ' -f2
    else
        echo "ERROR"
    fi
}

# Function to get dataset name
get_dataset_name() {
    basename "$1" .inp
}

# ============================================
# TESTING FUNCTIONS
# ============================================

# Test sequential implementation
test_sequential() {
    local dataset="$1"
    local dataset_name=$(get_dataset_name "$dataset")
    
    echo "=== Testing Sequential Implementation ==="
    echo "Dataset: $dataset"
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "  Run $run/$NUM_RUNS..."
        
        output_file="results/seq_${dataset_name}_run${run}.out"
        timing_file="${output_file}.timing"
        
        # Remove old timing file
        rm -f "$timing_file"
        
        # Run sequential version
        ./build/KMEANS_seq "$dataset" $K_CLUSTERS $MAX_ITERATIONS $THRESHOLD $TOLERANCE "$output_file" $SEED
        
        # Extract timing
        comp_time=$(extract_timing "$timing_file")
        if [ "$comp_time" != "ERROR" ]; then
            echo "$dataset_name,$run,$comp_time" >> logs/timing_logs/sequential_times.log
            echo "    Computation time: $comp_time seconds"
        else
            echo "    ERROR: Could not extract timing from $timing_file"
        fi
    done
    echo ""
}

# Test OpenMP implementation
test_openmp() {
    local dataset="$1"
    local dataset_name=$(get_dataset_name "$dataset")
    
    echo "=== Testing OpenMP Implementation ==="
    echo "Dataset: $dataset"
    
    for threads in "${OMP_THREADS[@]}"; do
        if [ $threads -gt $MAX_CORES ]; then
            continue
        fi
        
        echo "  Testing with $threads threads..."
        
        # Set OpenMP environment
        export OMP_NUM_THREADS=$threads
        export OMP_PROC_BIND=spread
        export OMP_PLACES=cores
        export OMP_DYNAMIC=false
        export OMP_SCHEDULE=dynamic,64
        
        for run in $(seq 1 $NUM_RUNS); do
            echo "    Run $run/$NUM_RUNS..."
            
            output_file="results/omp_${dataset_name}_${threads}t_run${run}.out"
            timing_file="${output_file}.timing"
            
            # Remove old timing file
            rm -f "$timing_file"
            
            # Run OpenMP version
            ./build/KMEANS_omp "$dataset" $K_CLUSTERS $MAX_ITERATIONS $THRESHOLD $TOLERANCE "$output_file" $SEED $threads
            
            # Extract timing
            comp_time=$(extract_timing "$timing_file")
            if [ "$comp_time" != "ERROR" ]; then
                echo "$dataset_name,$threads,$run,$comp_time" >> logs/timing_logs/openmp_times.log
                echo "      Computation time: $comp_time seconds"
            else
                echo "      ERROR: Could not extract timing from $timing_file"
            fi
        done
    done
    echo ""
}

# Test MPI implementation
test_mpi() {
    local dataset="$1"
    local dataset_name=$(get_dataset_name "$dataset")
    
    echo "=== Testing MPI Implementation ==="
    echo "Dataset: $dataset"
    
    for processes in "${MPI_PROCESSES[@]}"; do
        if [ $processes -gt $MAX_CORES ]; then
            continue
        fi
        
        echo "  Testing with $processes processes..."
        
        for run in $(seq 1 $NUM_RUNS); do
            echo "    Run $run/$NUM_RUNS..."
            
            output_file="results/mpi_${dataset_name}_${processes}p_run${run}.out"
            timing_file="${output_file}.timing"
            
            # Remove old timing file
            rm -f "$timing_file"
            
            # Run MPI version
            mpirun -np $processes --oversubscribe ./build/KMEANS_mpi "$dataset" $K_CLUSTERS $MAX_ITERATIONS $THRESHOLD $TOLERANCE "$output_file" $SEED
            
            # Extract timing
            comp_time=$(extract_timing "$timing_file")
            if [ "$comp_time" != "ERROR" ]; then
                echo "$dataset_name,$processes,$run,$comp_time" >> logs/timing_logs/mpi_times.log
                echo "      Computation time: $comp_time seconds"
            else
                echo "      ERROR: Could not extract timing from $timing_file"
            fi
        done
    done
    echo ""
}

# Test MPI+OpenMP implementation
test_mpi_openmp() {
    local dataset="$1"
    local dataset_name=$(get_dataset_name "$dataset")
    
    echo "=== Testing MPI+OpenMP Implementation ==="
    echo "Dataset: $dataset"
    
    for config in "${MPI_OMP_CONFIGS[@]}"; do
        processes=$(echo $config | cut -d':' -f1)
        threads=$(echo $config | cut -d':' -f2)
        total_cores=$((processes * threads))
        
        # Skip configurations that exceed available cores or max 4 processes
        if [ $total_cores -gt $MAX_CORES ] || [ $processes -gt 4 ]; then
            continue
        fi
        
        echo "  Testing with $processes processes × $threads threads (total: $total_cores cores)..."
        
        # Set OpenMP environment
        export OMP_NUM_THREADS=$threads
        export OMP_PROC_BIND=true
        export OMP_PLACES=cores
        export OMP_DYNAMIC=false
        export OMP_SCHEDULE=dynamic,64
        
        for run in $(seq 1 $NUM_RUNS); do
            echo "    Run $run/$NUM_RUNS..."
            
            output_file="results/mpi_omp_${dataset_name}_${processes}p_${threads}t_run${run}.out"
            timing_file="${output_file}.timing"
            
            # Remove old timing file
            rm -f "$timing_file"
            
            # Run MPI+OpenMP version
            mpirun -np $processes --oversubscribe ./build/KMEANS_mpi_omp "$dataset" $K_CLUSTERS $MAX_ITERATIONS $THRESHOLD $TOLERANCE "$output_file" $SEED $threads
            
            # Extract timing
            comp_time=$(extract_timing "$timing_file")
            if [ "$comp_time" != "ERROR" ]; then
                echo "$dataset_name,$processes,$threads,$total_cores,$run,$comp_time" >> logs/timing_logs/mpi_openmp_times.log
                echo "      Computation time: $comp_time seconds"
            else
                echo "      ERROR: Could not extract timing from $timing_file"
            fi
        done
    done
    echo ""
}

# ============================================
# MAIN TESTING LOOP
# ============================================

echo "=== Starting Comprehensive Testing ==="
echo "Datasets to test: ${#DATASETS[@]}"
echo "Runs per configuration: $NUM_RUNS"
echo "Maximum cores available: $MAX_CORES"
echo ""

for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$dataset" ]; then
        echo "WARNING: Dataset $dataset not found, skipping..."
        continue
    fi
    
    dataset_name=$(get_dataset_name "$dataset")
    echo "############################################"
    echo "TESTING DATASET: $dataset_name"
    echo "############################################"
    
    # Test each implementation
    test_sequential "$dataset"
    test_openmp "$dataset"
    test_mpi "$dataset"
    test_mpi_openmp "$dataset"
    
    echo "Completed testing for dataset: $dataset_name"
    echo ""
done

# ============================================
# SUMMARY
# ============================================

echo "============================================"
echo "=== COMPREHENSIVE TESTING COMPLETE ==="
echo "============================================"
echo "End time: $(date)"
echo ""
echo "Timing logs saved to:"
echo "  - Sequential: logs/timing_logs/sequential_times.log"
echo "  - OpenMP: logs/timing_logs/openmp_times.log"
echo "  - MPI: logs/timing_logs/mpi_times.log"
echo "  - MPI+OpenMP: logs/timing_logs/mpi_openmp_times.log"
echo ""
echo "Result files saved to: results/"
echo "Detailed output: logs/comprehensive_test_${SLURM_JOB_ID}.out"
echo ""

# Generate summary statistics
echo "=== SUMMARY STATISTICS ==="
for log_file in logs/timing_logs/*.log; do
    if [ -f "$log_file" ]; then
        implementation=$(basename "$log_file" .log | sed 's/_times//')
        total_tests=$(grep -v "^#" "$log_file" | grep -v "^dataset" | wc -l)
        echo "$implementation: $total_tests test runs completed"
    fi
done

echo ""
echo "All timing data collected from internal computation time measurements only."
echo "Use the log files for further analysis and visualization."
