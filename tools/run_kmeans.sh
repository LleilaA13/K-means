#!/bin/bash

> timing_log.txt

for i in 2 4 8 16 32 64
do
    echo "Run with $i threads"
    ./kmeans_omp test_files/input100D.inp 8 100 2.0 0.001 result_omp.txt 1 $i> /dev/null
done
