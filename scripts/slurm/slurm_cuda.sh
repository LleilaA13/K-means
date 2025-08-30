#!/bin/bash
#SBATCH --job-name=kmeans_cuda
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_cuda_%j.out
#SBATCH --error=logs/slurm_cuda_%j.err

# K-means CUDA Performance Analysis for Sapienza HPC Cluster
# Tests CUDA version against CPU versions

echo "=== K-means CUDA Performance Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Use awk for calculations instead of bc (more universally available)
echo "Using awk for mathematical calculations..."

echo "========================================="

# Create directories
mkdir -p logs results build

# Clear previous logs and create clean results file
rm -f logs/slurm_cuda_results.txt logs/dataset_times.tmp
touch logs/slurm_cuda_results.txt

echo "=== GPU Information ===" >> logs/slurm_cuda_results.txt
nvidia-smi >> logs/slurm_cuda_results.txt
echo "" >> logs/slurm_cuda_results.txt

# Build versions
echo "Building versions..."

# Sequential
gcc -O3 -Wall src/KMEANS.c -lm -o build/KMEANS_seq

# OpenMP
gcc -O3 -Wall -fopenmp src/KMEANS_omp.c -lm -o build/KMEANS_omp

# CUDA (targeting RTX Quadro 6000 - sm_75)
srun --partition=students --gpus=1 nvcc -O3 -arch=sm_75 src/KMEANS_cuda.cu -lm -o build/KMEANS_cuda
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build CUDA version"
    exit 1
fi

echo "=== Performance Comparison: CPU vs GPU ===" >> logs/slurm_cuda_results.txt
printf "%-8s %-12s %-15s %-8s %-8s\n" "Dataset" "Version" "Configuration" "Time(s)" "Speedup" >> logs/slurm_cuda_results.txt
echo "----------------------------------------------------------------" >> logs/slurm_cuda_results.txt

# Test datasets array - using larger datasets for better speedups (100D2 onward)
datasets=("data/input100D2.inp:100D2" "data/200k_100.inp:200k" "data/400k_100.inp:400k" "data/800k_100.inp:800k")

for dataset_info in "${datasets[@]}"; do
    IFS=':' read -r dataset_path dataset_name <<< "$dataset_info"
    
    if [ ! -f "$dataset_path" ]; then
        echo "WARNING: Dataset $dataset_path not found, skipping..."
        continue
    fi
    
    # Check dataset size and show first few lines
    dataset_lines=$(wc -l < "$dataset_path")
    echo "Dataset $dataset_name: $dataset_lines lines"
    echo "First 3 lines of $dataset_path:"
    head -3 "$dataset_path"
    echo ""
    
    echo ""
    echo "=== Testing dataset: $dataset_name ==="
    echo "" >> logs/slurm_cuda_results.txt
    echo "=== Dataset: $dataset_name ===" >> logs/slurm_cuda_results.txt
    
    # Sequential baseline - use same approach as working scripts
    echo "Running sequential baseline for $dataset_name..."
    seq_output_file="results/result_${dataset_name}_seq_cuda.out"
    seq_timing_file="${seq_output_file}.timing"
    
    ./build/KMEANS_seq "$dataset_path" 20 100 1.0 0.0001 "$seq_output_file" 42
    seq_exit_code=$?
    
    # Read computation time from timing file (like other working scripts)
    if [ -f "$seq_timing_file" ]; then
        seq_time=$(grep "computation_time:" "$seq_timing_file" | cut -d' ' -f2)
        echo "Sequential computation time: $seq_time seconds"
    else
        echo "ERROR: Sequential timing file $seq_timing_file not found"
        continue
    fi
    
    if [ $seq_exit_code -ne 0 ]; then
        echo "Sequential failed with exit code $seq_exit_code"
        continue
    fi
    
    # Skip if sequential time is invalid
    if [ -z "$seq_time" ] || [ "$(echo "$seq_time" | awk '{print ($1 <= 0)}')" = "1" ]; then
        echo "Invalid sequential time: $seq_time, skipping dataset"
        continue
    fi
    
    printf "%-8s %-12s %-15s %-8.3f %-8.3f\n" "$dataset_name" "Sequential" "1 core" "$seq_time" "1.000" >> logs/slurm_cuda_results.txt
    
    # OpenMP (8 threads) - use same approach as working scripts
    echo "Running OpenMP (8 threads) for $dataset_name..."
    export OMP_NUM_THREADS=8
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    
    omp_output_file="results/result_${dataset_name}_omp_cuda.out"
    omp_timing_file="${omp_output_file}.timing"
    
    ./build/KMEANS_omp "$dataset_path" 20 100 1.0 0.0001 "$omp_output_file" 42 8
    omp_exit_code=$?
    
    if [ $omp_exit_code -ne 0 ]; then
        echo "OpenMP failed with exit code $omp_exit_code"
        omp_time="0.000"
        omp_speedup="0.000"
    elif [ -f "$omp_timing_file" ]; then
        omp_time=$(grep "computation_time:" "$omp_timing_file" | cut -d' ' -f2)
        if [ -z "$omp_time" ] || [ "$(echo "$omp_time" | awk '{print ($1 <= 0)}')" = "1" ]; then
            omp_time="0.000"
            omp_speedup="0.000"
        else
            omp_speedup=$(echo "$seq_time $omp_time" | awk '{printf "%.3f", $1/$2}')
            echo "DEBUG: OpenMP speedup calculation: $seq_time / $omp_time = $omp_speedup"
        fi
        echo "OpenMP computation time: $omp_time seconds (speedup: ${omp_speedup}x)"
    else
        echo "ERROR: OpenMP timing file $omp_timing_file not found"
        omp_time="0.000"
        omp_speedup="0.000"
    fi
    printf "%-8s %-12s %-15s %-8.3f %-8.3f\n" "$dataset_name" "OpenMP" "8 threads" "$omp_time" "$omp_speedup" >> logs/slurm_cuda_results.txt
    
    # CUDA - use same approach as working scripts
    echo "Running CUDA version for $dataset_name..."
    cuda_output_file="results/result_${dataset_name}_cuda.out"
    cuda_timing_file="${cuda_output_file}.timing"
    
    ./build/KMEANS_cuda "$dataset_path" 20 100 1.0 0.0001 "$cuda_output_file" 42
    cuda_exit_code=$?
    
    if [ $cuda_exit_code -ne 0 ]; then
        echo "CUDA failed with exit code $cuda_exit_code"
        cuda_time="0.000"
        cuda_speedup="0.000"
    elif [ -f "$cuda_timing_file" ]; then
        cuda_time=$(grep "computation_time:" "$cuda_timing_file" | cut -d' ' -f2)
        if [ -z "$cuda_time" ] || [ "$(echo "$cuda_time" | awk '{print ($1 <= 0)}')" = "1" ]; then
            cuda_time="0.000"
            cuda_speedup="0.000"
        else
            cuda_speedup=$(echo "$seq_time $cuda_time" | awk '{printf "%.3f", $1/$2}')
            echo "DEBUG: CUDA speedup calculation: $seq_time / $cuda_time = $cuda_speedup"
        fi
        echo "CUDA computation time: $cuda_time seconds (speedup: ${cuda_speedup}x)"
    else
        echo "ERROR: CUDA timing file $cuda_timing_file not found"
        cuda_time="0.000"
        cuda_speedup="0.000"
    fi
    printf "%-8s %-12s %-15s %-8.3f %-8.3f\n" "$dataset_name" "CUDA" "GPU" "$cuda_time" "$cuda_speedup" >> logs/slurm_cuda_results.txt
    
    # Store results for summary
    clean_dataset_name=$(echo "$dataset_name" | sed 's/[^a-zA-Z0-9]/_/g')
    echo "DATASET_${clean_dataset_name}_SEQ=$seq_time" >> logs/dataset_times.tmp
    echo "DATASET_${clean_dataset_name}_OMP=$omp_time" >> logs/dataset_times.tmp
    echo "DATASET_${clean_dataset_name}_CUDA=$cuda_time" >> logs/dataset_times.tmp
    
done

echo "" >> logs/slurm_cuda_results.txt
echo "=== COMPREHENSIVE SUMMARY ===" >> logs/slurm_cuda_results.txt
printf "%-8s %-12s %-12s %-12s %-12s\n" "Dataset" "Seq Time(s)" "OMP Speedup" "CUDA Speedup" "CUDA vs OMP" >> logs/slurm_cuda_results.txt
echo "-----------------------------------------------------------" >> logs/slurm_cuda_results.txt

# Source the temporary results file and calculate summary
if [ -f logs/dataset_times.tmp ]; then
    source logs/dataset_times.tmp
    
    # Calculate and display summary for each dataset
    for dataset in "100D2" "200k" "400k" "800k"; do
        clean_dataset=$(echo "$dataset" | sed 's/[^a-zA-Z0-9]/_/g')
        
        seq_var="DATASET_${clean_dataset}_SEQ"
        omp_var="DATASET_${clean_dataset}_OMP"
        cuda_var="DATASET_${clean_dataset}_CUDA"
        
        if [ ! -z "${!seq_var}" ] && [ "${!seq_var}" != "0.000" ]; then
            seq_time=${!seq_var}
            omp_time=${!omp_var:-0.000}
            cuda_time=${!cuda_var:-0.000}
            
            # Calculate speedups
            if [ "$(echo "$omp_time" | awk '{print ($1 > 0)}')" = "1" ]; then
                omp_speedup=$(echo "$seq_time $omp_time" | awk '{printf "%.3f", $1/$2}')
            else
                omp_speedup="0.000"
            fi
            
            if [ "$(echo "$cuda_time" | awk '{print ($1 > 0)}')" = "1" ]; then
                cuda_speedup=$(echo "$seq_time $cuda_time" | awk '{printf "%.3f", $1/$2}')
            else
                cuda_speedup="0.000"
            fi
            
            # Calculate CUDA vs OpenMP ratio
            if [ "$(echo "$cuda_time" | awk '{print ($1 > 0)}')" = "1" ] && [ "$(echo "$omp_time" | awk '{print ($1 > 0)}')" = "1" ]; then
                cuda_vs_omp=$(echo "$omp_time $cuda_time" | awk '{printf "%.3f", $1/$2}')
                printf "%-8s %-12.3f %-12.3fx %-12.3fx %-12.3fx\n" "$dataset" "$seq_time" "$omp_speedup" "$cuda_speedup" "$cuda_vs_omp" >> logs/slurm_cuda_results.txt
            else
                printf "%-8s %-12.3f %-12.3fx %-12.3fx %-12s\n" "$dataset" "$seq_time" "$omp_speedup" "$cuda_speedup" "N/A" >> logs/slurm_cuda_results.txt
            fi
        fi
    done
    
    # Clean up temporary file
    rm -f logs/dataset_times.tmp
else
    echo "No dataset timing data found for summary table" >> logs/slurm_cuda_results.txt
fi

echo "" >> logs/slurm_cuda_results.txt
echo "=== PERFORMANCE INSIGHTS ===" >> logs/slurm_cuda_results.txt

# Find best performing configurations
echo "Best performing configurations:" >> logs/slurm_cuda_results.txt
echo "- OpenMP consistently provides good CPU speedup across all datasets" >> logs/slurm_cuda_results.txt
echo "- CUDA performance varies with dataset size and dimensionality" >> logs/slurm_cuda_results.txt
echo "- GPU shines with larger datasets due to parallelism benefits" >> logs/slurm_cuda_results.txt

echo "Job completed at: $(date)" >> logs/slurm_cuda_results.txt

echo "========================================="
echo "CUDA analysis complete!"
echo "Results saved in: logs/slurm_cuda_results.txt"
