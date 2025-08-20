#!/bin/bash
#SBATCH --job-name=kmeans_all_versions
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_comparison_%j.out
#SBATCH --error=logs/slurm_comparison_%j.err

# Comprehensive K-means Performance Comparison for SLURM
# Tests Sequential, OpenMP, and optionally MPI versions

echo "=== K-means Comprehensive Performance Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Cores available: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "================================================="

# Create directories
mkdir -p logs results build

# Clear previous logs
> logs/timing_log_omp.txt
> logs/timing_log_seq.txt
> logs/slurm_comprehensive_results.txt

echo "=== System Information ===" >> logs/slurm_comprehensive_results.txt
echo "Node: $SLURM_NODELIST" >> logs/slurm_comprehensive_results.txt
echo "Date: $(date)" >> logs/slurm_comprehensive_results.txt
echo "Available cores: $SLURM_CPUS_PER_TASK" >> logs/slurm_comprehensive_results.txt
echo "" >> logs/slurm_comprehensive_results.txt

# Build all versions
echo "Building all versions..."

# Sequential version
gcc -O3 -Wall src/KMEANS.c -lm -o build/KMEANS_seq
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build sequential version"
    exit 1
fi

# OpenMP version
gcc -O3 -Wall -fopenmp src/KMEANS_omp.c -lm -o build/KMEANS_omp
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build OpenMP version"
    exit 1
fi

# MPI version (if available)
if command -v mpicc &> /dev/null; then
    mpicc -O3 -Wall src/KMEANS_mpi.c -lm -o build/KMEANS_mpi
    mpi_available=true
    echo "MPI version built successfully"
else
    mpi_available=false
    echo "MPI compiler not available, skipping MPI tests"
fi

# Test datasets (use different sizes if available)
test_datasets=(
    "data/input100D.inp:100D"
    "data/input20D.inp:20D"
    "data/input10D.inp:10D"
    "data/input2D.inp:2D"
)

echo "=== Performance Comparison Results ===" >> logs/slurm_comprehensive_results.txt
echo "Dataset\tVersion\tThreads/Procs\tTime(s)\tSpeedup\tEfficiency(%)" >> logs/slurm_comprehensive_results.txt
echo "--------------------------------------------------------------------" >> logs/slurm_comprehensive_results.txt

for dataset_entry in "${test_datasets[@]}"; do
    IFS=':' read -r dataset_path dataset_name <<< "$dataset_entry"
    
    if [ ! -f "$dataset_path" ]; then
        echo "Skipping $dataset_name - file not found: $dataset_path"
        continue
    fi
    
    echo "Testing dataset: $dataset_name ($dataset_path)"
    
    # Sequential baseline
    echo "  Running sequential version..."
    start_time=$(date +%s.%N)
    ./build/KMEANS_seq "$dataset_path" 20 100 1.0 0.0001 "results/result_${dataset_name}_seq_slurm.out" 42
    end_time=$(date +%s.%N)
    seq_time=$(echo "$end_time - $start_time" | bc -l)
    
    echo "$dataset_name\tSequential\t1\t$seq_time\t1.000\t100.0" >> logs/slurm_comprehensive_results.txt
    
    # OpenMP versions
    for threads in 2 4 8; do
        echo "  Running OpenMP with $threads threads..."
        export OMP_NUM_THREADS=$threads
        export OMP_PROC_BIND=true
        export OMP_PLACES=cores
        
        start_time=$(date +%s.%N)
        ./build/KMEANS_omp "$dataset_path" 20 100 1.0 0.0001 "results/result_${dataset_name}_omp_${threads}t_slurm.out" 42 $threads
        end_time=$(date +%s.%N)
        omp_time=$(echo "$end_time - $start_time" | bc -l)
        
        speedup=$(echo "scale=3; $seq_time / $omp_time" | bc -l)
        efficiency=$(echo "scale=1; $speedup * 100 / $threads" | bc -l)
        
        echo "$dataset_name\tOpenMP\t$threads\t$omp_time\t$speedup\t$efficiency" >> logs/slurm_comprehensive_results.txt
    done
    
    # MPI versions (if available)
    if [ "$mpi_available" = true ]; then
        for procs in 2 4 8; do
            echo "  Running MPI with $procs processes..."
            start_time=$(date +%s.%N)
            mpirun -np $procs ./build/KMEANS_mpi "$dataset_path" 20 100 1.0 0.0001 "results/result_${dataset_name}_mpi_${procs}p_slurm.out" 42
            end_time=$(date +%s.%N)
            mpi_time=$(echo "$end_time - $start_time" | bc -l)
            
            speedup=$(echo "scale=3; $seq_time / $mpi_time" | bc -l)
            efficiency=$(echo "scale=1; $speedup * 100 / $procs" | bc -l)
            
            echo "$dataset_name\tMPI\t$procs\t$mpi_time\t$speedup\t$efficiency" >> logs/slurm_comprehensive_results.txt
        done
    fi
    
    echo "" >> logs/slurm_comprehensive_results.txt
done

echo "" >> logs/slurm_comprehensive_results.txt
echo "=== Analysis Summary ===" >> logs/slurm_comprehensive_results.txt

# Find best configurations for each dataset
echo "Best configurations per dataset:" >> logs/slurm_comprehensive_results.txt
for dataset_entry in "${test_datasets[@]}"; do
    IFS=':' read -r dataset_path dataset_name <<< "$dataset_entry"
    
    if [ ! -f "$dataset_path" ]; then
        continue
    fi
    
    best_line=$(grep "^$dataset_name" logs/slurm_comprehensive_results.txt | grep -v "Sequential" | sort -k5 -nr | head -n 1)
    if [ ! -z "$best_line" ]; then
        version=$(echo "$best_line" | cut -f2)
        threads=$(echo "$best_line" | cut -f3)
        speedup=$(echo "$best_line" | cut -f5)
        echo "  $dataset_name: $version with $threads threads/procs (${speedup}x speedup)" >> logs/slurm_comprehensive_results.txt
    fi
done

echo "" >> logs/slurm_comprehensive_results.txt
echo "Job completed at: $(date)" >> logs/slurm_comprehensive_results.txt

echo "================================================="
echo "Comprehensive analysis complete!"
echo "Results saved in: logs/slurm_comprehensive_results.txt"
echo "SLURM output: logs/slurm_comparison_${SLURM_JOB_ID}.out"
