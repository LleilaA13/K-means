#!/bin/bash
#SBATCH --job-name=kmeans_cuda
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_cuda_%j.out
#SBATCH --error=logs/slurm_cuda_%j.err

# K-means CUDA Performance Analysis for Sapienza HPC Cluster
# Tests CUDA version against CPU versions

echo "=== K-means CUDA Performance Analysis ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Create directories
mkdir -p logs results build

# Clear previous logs
> logs/slurm_cuda_results.txt

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
nvcc -O3 -arch=sm_75 src/KMEANS_cuda.cu -o build/KMEANS_cuda
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build CUDA version"
    exit 1
fi

echo "=== Performance Comparison: CPU vs GPU ===" >> logs/slurm_cuda_results.txt
echo "Version\tConfiguration\tTime(s)\tSpeedup" >> logs/slurm_cuda_results.txt
echo "----------------------------------------" >> logs/slurm_cuda_results.txt

# Test with 100D dataset
dataset="data/input100D.inp"
if [ ! -f "$dataset" ]; then
    echo "ERROR: Dataset $dataset not found"
    exit 1
fi

# Sequential baseline
echo "Running sequential baseline..."
start_time=$(date +%s.%N)
./build/KMEANS_seq "$dataset" 20 100 1.0 0.0001 results/result_cuda_seq.out 42
end_time=$(date +%s.%N)
seq_time=$(echo "$end_time - $start_time" | bc -l)
echo "Sequential\t1 core\t$seq_time\t1.000" >> logs/slurm_cuda_results.txt

# OpenMP (best configuration from previous tests)
echo "Running OpenMP (4 threads)..."
export OMP_NUM_THREADS=4
export OMP_PROC_BIND=true
export OMP_PLACES=cores
start_time=$(date +%s.%N)
./build/KMEANS_omp "$dataset" 20 100 1.0 0.0001 results/result_cuda_omp.out 42 4
end_time=$(date +%s.%N)
omp_time=$(echo "$end_time - $start_time" | bc -l)
omp_speedup=$(echo "scale=3; $seq_time / $omp_time" | bc -l)
echo "OpenMP\t4 threads\t$omp_time\t$omp_speedup" >> logs/slurm_cuda_results.txt

# CUDA
echo "Running CUDA version..."
start_time=$(date +%s.%N)
./build/KMEANS_cuda "$dataset" 20 100 1.0 0.0001 results/result_cuda_gpu.out 42
end_time=$(date +%s.%N)
cuda_time=$(echo "$end_time - $start_time" | bc -l)
cuda_speedup=$(echo "scale=3; $seq_time / $cuda_time" | bc -l)
echo "CUDA\tGPU\t$cuda_time\t$cuda_speedup" >> logs/slurm_cuda_results.txt

echo "" >> logs/slurm_cuda_results.txt
echo "=== Summary ===" >> logs/slurm_cuda_results.txt
echo "Sequential time: $seq_time seconds" >> logs/slurm_cuda_results.txt
echo "OpenMP speedup: ${omp_speedup}x" >> logs/slurm_cuda_results.txt
echo "CUDA speedup: ${cuda_speedup}x" >> logs/slurm_cuda_results.txt

# Compare CUDA vs OpenMP
cuda_vs_omp=$(echo "scale=3; $omp_time / $cuda_time" | bc -l)
echo "CUDA vs OpenMP: ${cuda_vs_omp}x faster" >> logs/slurm_cuda_results.txt

echo "Job completed at: $(date)" >> logs/slurm_cuda_results.txt

echo "========================================="
echo "CUDA analysis complete!"
echo "Results saved in: logs/slurm_cuda_results.txt"
