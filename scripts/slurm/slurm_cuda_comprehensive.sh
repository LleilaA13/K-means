#!/bin/bash
#SBATCH --job-name=kmeans_cuda_comprehensive
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/cuda_comprehensive_%j.out
#SBATCH --error=logs/cuda_comprehensive_%j.err

# Comprehensive K-means CUDA Performance Testing Script
# Tests CUDA implementation across all larger datasets with multiple runs
# Matches the comprehensive testing approach but focuses only on GPU acceleration
# Author: Automated CUDA testing script
# Date: $(date)

echo "=== Comprehensive K-means CUDA Performance Testing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "============================================"

# ============================================
# CONFIGURATION
# ============================================

# Datasets to test (matching comprehensive script)
DATASETS=(
    "data/input100D.inp"
    "data/input100D2.inp"
    "data/200k_100.inp" 
    "data/400k_100.inp"
    "data/800k_100.inp"
)

# Number of test runs per configuration (matching comprehensive script)
NUM_RUNS=5

# Standard K-means parameters (matching comprehensive script)
K_CLUSTERS=20
MAX_ITERATIONS=3000
THRESHOLD=1.0
TOLERANCE=0.0001
SEED=42

# ============================================
# SETUP
# ============================================

# Create necessary directories
mkdir -p logs results logs/timing_logs build

# Display GPU information
echo "=== GPU Information ==="
nvidia-smi
echo ""

# Initialize CUDA timing log file
echo "# CUDA K-means Timing Results" > logs/timing_logs/cuda_times.log
echo "# Format: dataset,run,computation_time_seconds" >> logs/timing_logs/cuda_times.log
echo "dataset,run,computation_time" >> logs/timing_logs/cuda_times.log

# ============================================
# BUILD CUDA EXECUTABLE
# ============================================

echo "=== Building CUDA Executable ==="

# Build CUDA version (targeting RTX Quadro 6000 - sm_75, same as original cuda script)
if [ ! -f "build/KMEANS_cuda" ]; then
    echo "Building CUDA version..."
    nvcc -arch=sm_75 src/KMEANS_cuda.cu -lm -o build/KMEANS_cuda
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to build CUDA version"
        exit 1
    fi
    echo "CUDA executable built successfully."
else
    echo "CUDA executable already exists."
fi

# Verify CUDA executable
if [ ! -x "build/KMEANS_cuda" ]; then
    echo "ERROR: CUDA executable not found or not executable"
    exit 1
fi

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
# CUDA TESTING FUNCTION
# ============================================

# Test CUDA implementation
test_cuda() {
    local dataset="$1"
    local dataset_name=$(get_dataset_name "$dataset")
    
    echo "=== Testing CUDA Implementation ==="
    echo "Dataset: $dataset"
    
    # Check dataset size and show first few lines (for debugging)
    if [ -f "$dataset" ]; then
        dataset_lines=$(wc -l < "$dataset")
        echo "Dataset $dataset_name: $dataset_lines lines"
        echo "First 3 lines of $dataset:"
        head -3 "$dataset"
    else
        echo "ERROR: Dataset $dataset not found"
        return 1
    fi
    
    for run in $(seq 1 $NUM_RUNS); do
        echo "  Run $run/$NUM_RUNS..."
        
        output_file="results/cuda_${dataset_name}_run${run}.out"
        timing_file="${output_file}.timing"
        
        # Remove old timing file
        rm -f "$timing_file"
        
        # Run CUDA version (using same parameters as comprehensive script)
        ./build/KMEANS_cuda "$dataset" $K_CLUSTERS $MAX_ITERATIONS $THRESHOLD $TOLERANCE "$output_file" $SEED
        cuda_exit_code=$?
        
        if [ $cuda_exit_code -ne 0 ]; then
            echo "    ERROR: CUDA execution failed with exit code $cuda_exit_code"
            echo "$dataset_name,$run,ERROR" >> logs/timing_logs/cuda_times.log
            continue
        fi
        
        # Extract timing
        comp_time=$(extract_timing "$timing_file")
        if [ "$comp_time" != "ERROR" ] && [ ! -z "$comp_time" ]; then
            echo "$dataset_name,$run,$comp_time" >> logs/timing_logs/cuda_times.log
            echo "    Computation time: $comp_time seconds"
            
            # Validate timing is reasonable (positive number)
            if [ "$(echo "$comp_time" | awk '{print ($1 > 0)}')" = "1" ]; then
                echo "    ✓ Valid timing recorded"
            else
                echo "    ⚠ Warning: Invalid timing value: $comp_time"
            fi
        else
            echo "    ERROR: Could not extract timing from $timing_file"
            echo "$dataset_name,$run,ERROR" >> logs/timing_logs/cuda_times.log
        fi
        
        # Brief pause between runs
        sleep 1
    done
    echo ""
}

# ============================================
# MAIN TESTING LOOP
# ============================================

echo "=== Starting Comprehensive CUDA Testing ==="
echo "Datasets to test: ${#DATASETS[@]}"
echo "Runs per dataset: $NUM_RUNS"
echo "Total CUDA tests: $((${#DATASETS[@]} * NUM_RUNS))"
echo ""

successful_datasets=0
total_tests=0
successful_tests=0

for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$dataset" ]; then
        echo "WARNING: Dataset $dataset not found, skipping..."
        continue
    fi
    
    dataset_name=$(get_dataset_name "$dataset")
    echo "############################################"
    echo "TESTING DATASET: $dataset_name"
    echo "############################################"
    
    # Test CUDA implementation
    test_cuda "$dataset"
    
    # Count successful tests for this dataset
    dataset_successful=$(grep "^$dataset_name," logs/timing_logs/cuda_times.log | grep -v "ERROR" | wc -l)
    if [ $dataset_successful -gt 0 ]; then
        successful_datasets=$((successful_datasets + 1))
        successful_tests=$((successful_tests + dataset_successful))
    fi
    total_tests=$((total_tests + NUM_RUNS))
    
    echo "Completed CUDA testing for dataset: $dataset_name ($dataset_successful/$NUM_RUNS successful)"
    echo ""
done

# ============================================
# SUMMARY AND STATISTICS
# ============================================

echo "============================================"
echo "=== COMPREHENSIVE CUDA TESTING COMPLETE ==="
echo "============================================"
echo "End time: $(date)"
echo ""

echo "=== TESTING SUMMARY ==="
echo "Datasets tested: $successful_datasets/${#DATASETS[@]}"
echo "Total test runs: $successful_tests/$total_tests successful"
echo "Success rate: $(echo "$successful_tests $total_tests" | awk '{printf "%.1f%%", ($1/$2)*100}')"
echo ""

echo "CUDA timing log saved to: logs/timing_logs/cuda_times.log"
echo "Result files saved to: results/"
echo "Detailed output: logs/cuda_comprehensive_${SLURM_JOB_ID}.out"
echo ""

# Generate detailed statistics from timing log
echo "=== DETAILED CUDA PERFORMANCE STATISTICS ==="
if [ -f "logs/timing_logs/cuda_times.log" ]; then
    echo "Dataset-wise CUDA performance summary:"
    echo "----------------------------------------"
    printf "%-12s %-8s %-10s %-10s %-10s %-10s\n" "Dataset" "Runs" "Avg(s)" "Min(s)" "Max(s)" "StdDev"
    echo "----------------------------------------------------------------"
    
    for dataset in "${DATASETS[@]}"; do
        dataset_name=$(get_dataset_name "$dataset")
        
        # Extract successful times for this dataset
        times=$(grep "^$dataset_name," logs/timing_logs/cuda_times.log | grep -v "ERROR" | cut -d',' -f3)
        
        if [ ! -z "$times" ]; then
            run_count=$(echo "$times" | wc -l)
            avg_time=$(echo "$times" | awk '{sum+=$1} END {if(NR>0) printf "%.3f", sum/NR; else print "0.000"}')
            min_time=$(echo "$times" | awk 'BEGIN{min=999999} {if($1<min) min=$1} END {printf "%.3f", min}')
            max_time=$(echo "$times" | awk 'BEGIN{max=0} {if($1>max) max=$1} END {printf "%.3f", max}')
            
            # Calculate standard deviation
            std_dev=$(echo "$times" | awk -v avg="$avg_time" '{sum+=($1-avg)^2} END {if(NR>1) printf "%.3f", sqrt(sum/(NR-1)); else print "0.000"}')
            
            printf "%-12s %-8d %-10s %-10s %-10s %-10s\n" "$dataset_name" "$run_count" "$avg_time" "$min_time" "$max_time" "$std_dev"
        else
            printf "%-12s %-8s %-10s %-10s %-10s %-10s\n" "$dataset_name" "0" "N/A" "N/A" "N/A" "N/A"
        fi
    done
    
    echo ""
    echo "=== CUDA PERFORMANCE INSIGHTS ==="
    echo "- All timing measurements are from internal CUDA computation time only"
    echo "- Multiple runs per dataset provide statistical reliability"
    echo "- Results can be used for comparison with CPU implementations"
    echo "- CUDA performance typically improves with larger, higher-dimensional datasets"
    
    # Find best and worst performing datasets
    best_dataset=""
    best_time=999999
    worst_dataset=""
    worst_time=0
    
    for dataset in "${DATASETS[@]}"; do
        dataset_name=$(get_dataset_name "$dataset")
        times=$(grep "^$dataset_name," logs/timing_logs/cuda_times.log | grep -v "ERROR" | cut -d',' -f3)
        
        if [ ! -z "$times" ]; then
            avg_time=$(echo "$times" | awk '{sum+=$1} END {if(NR>0) print sum/NR; else print 0}')
            
            if [ "$(echo "$avg_time < $best_time" | bc -l 2>/dev/null || echo "$avg_time" | awk -v bt="$best_time" '{print ($1 < bt)}')" = "1" ]; then
                best_time=$avg_time
                best_dataset=$dataset_name
            fi
            
            if [ "$(echo "$avg_time > $worst_time" | bc -l 2>/dev/null || echo "$avg_time" | awk -v wt="$worst_time" '{print ($1 > wt)}')" = "1" ]; then
                worst_time=$avg_time
                worst_dataset=$dataset_name
            fi
        fi
    done
    
    if [ ! -z "$best_dataset" ]; then
        echo "- Fastest CUDA performance: $best_dataset (avg: ${best_time}s)"
    fi
    if [ ! -z "$worst_dataset" ]; then
        echo "- Slowest CUDA performance: $worst_dataset (avg: ${worst_time}s)"
    fi
    
else
    echo "No CUDA timing data found for analysis"
fi

echo ""
echo "Use logs/timing_logs/cuda_times.log for further analysis and comparison."
echo "This data can be processed by the comprehensive_performance_analysis.py script."
echo ""
echo "CUDA comprehensive testing completed successfully!"
 
# Usage: sbatch scripts/slurm/slurm_cuda_comprehensive.sh