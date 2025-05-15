#!/bin/bash

> timing_log.txt

for i in {1..20}
do
    echo "Run $i"
    ./kmeans_omp test_files/input100D.inp 8 100 2.0 0.001 result_omp.txt $i > /dev/null
done
