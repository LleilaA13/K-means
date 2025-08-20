#!/bin/bash
#SBATCH --job-name=kmeans_python_analysis
#SBATCH --partition=students
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_python_%j.out
#SBATCH --error=logs/slurm_python_%j.err

# K-means Python Analysis Script for SLURM
# Runs the comprehensive Python analysis tool

echo "=== K-means Python Analysis on SLURM ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================"

# Create directories
mkdir -p logs results

# Load Python module if available (common on HPC clusters)
if command -v module &> /dev/null; then
    echo "Loading Python module..."
    module load python/3.8 2>/dev/null || module load python3 2>/dev/null || echo "No Python module found, using system Python"
fi

# Check Python availability
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python not found"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Run the cluster analysis
echo "Starting comprehensive K-means analysis..."
$PYTHON_CMD scripts/cluster_analysis.py

echo "========================================"
echo "Python analysis completed at: $(date)"
echo "Check logs/cluster_performance_report.txt for results"
