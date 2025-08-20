#!/bin/bash

echo "Starting KMeans benchmark"

# Ensure we're in the project root directory
cd "$(dirname "$0")/.."

# Create necessary directories
mkdir -p results logs

for i in 2 4 8 16 32 64
do
    echo "Running with $i threads..."
    ./../../build/KMEANS_omp ./../../data/input100D.inp 20 100 1.0 0.0001 ../../results/result_omp_${i}threads.txt 1 $i
done

echo "Benchmark complete!"
echo "Results saved in ../../results/ directory"
echo "Timing logs saved in ../../logs/ directory"
