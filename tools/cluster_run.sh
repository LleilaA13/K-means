#!/bin/bash
for i in 1 2 3 4 5 6
do
    threads=$((2**i))
    srun --partition=multicore --cpus-per-task=$threads ./kmeans_omp test_files/input100D2.inp 20 100 1.0 0.0001 result_omp 1 $threads
done
