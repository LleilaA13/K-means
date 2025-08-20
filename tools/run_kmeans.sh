#!/bin/bash

echo "Starting KMeans benchmark"

for i in 2 4 8 16 32 64
do
    echo "Running with $i threads..."
    ./kmeans_omp ./input100D2.inp 20 100 1.0 0.0001 result_omp.txt 1 $i
done

echo "Benchmark complete!"
