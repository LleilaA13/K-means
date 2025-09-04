#!/bin/bash
#SBATCH --job-name=clustering_consistency
#SBATCH --partition=students
#SBATCH --time=00:30:00
#SBATCH --output=logs/clustering_test_%j.out
#SBATCH --error=logs/clustering_test_%j.err

# Script to test clustering consistency across all implementations
# Same seed and parameters for all versions

# Configuration variables
INPUT_FILE="data/input100D2.inp"
CLUSTERS=20
MAX_ITERATIONS=3000
MIN_CHANGES=1.0
THRESHOLD=0.0001
SEED=42
OMP_THREADS=8
MPI_PROCS_HYBRID=2
OMP_THREADS_HYBRID=4
MPI_PROCS=8

# Output files
SEQ_OUTPUT="results/test_seq_100D2.out"
OMP_OUTPUT="results/test_omp_100D2.out"
MPI_OMP_OUTPUT="results/test_mpi_omp_100D2.out"
MPI_OUTPUT="results/test_mpi_100D2.out"
CUDA_OUTPUT="results/test_cuda_100D2.out"

echo "Testing clustering consistency with ${INPUT_FILE}..."
echo "=============================================="
echo "Parameters: K=${CLUSTERS}, iterations=${MAX_ITERATIONS}, changes=${MIN_CHANGES}, threshold=${THRESHOLD}, seed=${SEED}"
echo ""

echo "Running Sequential implementation..."
srun --partition=students ./build/KMEANS_seq ${INPUT_FILE} ${CLUSTERS} ${MAX_ITERATIONS} ${MIN_CHANGES} ${THRESHOLD} ${SEQ_OUTPUT} ${SEED}

echo "Running OpenMP implementation (${OMP_THREADS} threads)..."
srun --partition=students --cpus-per-task=${OMP_THREADS} ./build/KMEANS_omp ${INPUT_FILE} ${CLUSTERS} ${MAX_ITERATIONS} ${MIN_CHANGES} ${THRESHOLD} ${OMP_OUTPUT} ${SEED} ${OMP_THREADS}

echo "Running MPI+OpenMP implementation (${MPI_PROCS_HYBRID} MPI × ${OMP_THREADS_HYBRID} OMP)..."
srun --partition=students --nodes=1 --cpus-per-task=$((MPI_PROCS_HYBRID * OMP_THREADS_HYBRID)) mpirun -np ${MPI_PROCS_HYBRID} --oversubscribe ./build/KMEANS_mpi_omp ${INPUT_FILE} ${CLUSTERS} ${MAX_ITERATIONS} ${MIN_CHANGES} ${THRESHOLD} ${MPI_OMP_OUTPUT} ${SEED} ${OMP_THREADS_HYBRID}

echo "Running MPI implementation (${MPI_PROCS} processes)..."
srun --partition=students --cpus-per-task=${OMP_THREADS} mpirun -np ${MPI_PROCS} --oversubscribe ./build/KMEANS_mpi ${INPUT_FILE} ${CLUSTERS} ${MAX_ITERATIONS} ${MIN_CHANGES} ${THRESHOLD} ${MPI_OUTPUT} ${SEED}

echo "Running CUDA implementation..."
srun --partition=students --gpus=1 ./build/KMEANS_cuda ${INPUT_FILE} ${CLUSTERS} ${MAX_ITERATIONS} ${MIN_CHANGES} ${THRESHOLD} ${CUDA_OUTPUT} ${SEED}

echo ""
echo "=============================================="
echo "All tests completed. Check results/ directory for output files."
echo "Sequential: ${SEQ_OUTPUT}"
echo "OpenMP: ${OMP_OUTPUT}"
echo "MPI+OpenMP: ${MPI_OMP_OUTPUT}"
echo "MPI: ${MPI_OUTPUT}"
echo "CUDA: ${CUDA_OUTPUT}"
