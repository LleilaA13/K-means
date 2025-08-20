#!/usr/bin/env python3
"""
SLURM Cluster Performance Analysis Tool for K-means
Adapted version of log_module.py specifically for cluster environments
"""

import os
import sys
import subprocess
import time
import re
from pathlib import Path

def run_command(cmd, timeout=300):
    """Run a command with timeout and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, 
                              text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"

def build_applications():
    """Build all K-means versions for cluster."""
    print("Building applications for cluster...")
    
    # Create build directory
    os.makedirs("../../build", exist_ok=True)
    
    build_commands = [
        ("Sequential", "gcc -O3 -Wall ../../src/KMEANS.c -lm -o ../../build/KMEANS_seq"),
        ("OpenMP", "gcc -O3 -Wall -fopenmp ../../src/KMEANS_omp.c -lm -o ../../build/KMEANS_omp"),
    ]
    
    # Check if MPI is available
    success, _, _ = run_command("which mpicc")
    if success:
        build_commands.append(("MPI", "mpicc -O3 -Wall ../../src/KMEANS_mpi.c -lm -o ../../build/KMEANS_mpi"))
    
    # Check if CUDA is available
    success, _, _ = run_command("which nvcc")
    if success:
        build_commands.append(("CUDA", "nvcc -O3 -arch=sm_75 ../../src/KMEANS_cuda.cu -o ../../build/KMEANS_cuda"))
    
    built_versions = []
    for version, cmd in build_commands:
        print(f"  Building {version}...")
        success, stdout, stderr = run_command(cmd)
        if success:
            built_versions.append(version)
            print(f"    ✓ {version} built successfully")
        else:
            print(f"    ✗ {version} build failed: {stderr}")
    
    return built_versions

def get_system_info():
    """Get system information for cluster analysis."""
    info = {}
    
    # SLURM job information
    info['job_id'] = os.environ.get('SLURM_JOB_ID', 'N/A')
    info['node'] = os.environ.get('SLURM_NODELIST', 'N/A')
    info['cpus'] = os.environ.get('SLURM_CPUS_PER_TASK', 'N/A')
    info['memory'] = os.environ.get('SLURM_MEM_PER_NODE', 'N/A')
    
    # CPU information
    success, output, _ = run_command("lscpu | grep 'Model name' | head -1")
    if success:
        info['cpu_model'] = output.split(':')[1].strip()
    else:
        info['cpu_model'] = 'Unknown'
    
    # Available cores
    success, output, _ = run_command("nproc")
    if success:
        info['total_cores'] = int(output.strip())
    else:
        info['total_cores'] = 1
    
    return info

def run_performance_test(dataset_path, dataset_name, built_versions):
    """Run performance tests for all available versions."""
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found, skipping...")
        return {}
    
    results = {}
    
    # Sequential baseline
    if "Sequential" in built_versions:
        print(f"  Running sequential baseline for {dataset_name}...")
        start_time = time.time()
        success, _, stderr = run_command(
            f"../../build/KMEANS_seq {dataset_path} 20 100 1.0 0.0001 "
            f"../../results/result_{dataset_name}_seq_cluster.out 42"
        )
        end_time = time.time()
        
        if success:
            seq_time = end_time - start_time
            results['sequential'] = {'time': seq_time, 'speedup': 1.0, 'efficiency': 100.0}
            print(f"    Sequential: {seq_time:.3f}s")
        else:
            print(f"    Sequential failed: {stderr}")
            return {}
    
    baseline_time = results.get('sequential', {}).get('time', 1.0)
    
    # OpenMP tests
    if "OpenMP" in built_versions:
        results['openmp'] = {}
        max_threads = min(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')), 8)
        
        for threads in [2, 4, max_threads]:
            if threads > max_threads:
                continue
            
            print(f"  Running OpenMP with {threads} threads...")
            
            # Set OpenMP environment
            os.environ['OMP_NUM_THREADS'] = str(threads)
            os.environ['OMP_PROC_BIND'] = 'true'
            os.environ['OMP_PLACES'] = 'cores'
            
            start_time = time.time()
            success, _, stderr = run_command(
                f"../../build/KMEANS_omp {dataset_path} 20 100 1.0 0.0001 "
                f"../../results/result_{dataset_name}_omp_{threads}t_cluster.out 42 {threads}"
            )
            end_time = time.time()
            
            if success:
                omp_time = end_time - start_time
                speedup = baseline_time / omp_time
                efficiency = (speedup / threads) * 100
                results['openmp'][threads] = {
                    'time': omp_time, 
                    'speedup': speedup, 
                    'efficiency': efficiency
                }
                print(f"    OpenMP {threads}t: {omp_time:.3f}s (speedup: {speedup:.2f}x)")
            else:
                print(f"    OpenMP {threads}t failed: {stderr}")
    
    # MPI tests
    if "MPI" in built_versions:
        results['mpi'] = {}
        max_procs = min(int(os.environ.get('SLURM_CPUS_PER_TASK', '8')), 8)
        
        for procs in [2, 4, max_procs]:
            if procs > max_procs:
                continue
            
            print(f"  Running MPI with {procs} processes...")
            
            start_time = time.time()
            success, _, stderr = run_command(
                f"mpirun -np {procs} ../../build/KMEANS_mpi {dataset_path} 20 100 1.0 0.0001 "
                f"../../results/result_{dataset_name}_mpi_{procs}p_cluster.out 42"
            )
            end_time = time.time()
            
            if success:
                mpi_time = end_time - start_time
                speedup = baseline_time / mpi_time
                efficiency = (speedup / procs) * 100
                results['mpi'][procs] = {
                    'time': mpi_time, 
                    'speedup': speedup, 
                    'efficiency': efficiency
                }
                print(f"    MPI {procs}p: {mpi_time:.3f}s (speedup: {speedup:.2f}x)")
            else:
                print(f"    MPI {procs}p failed: {stderr}")
    
    # CUDA test
    if "CUDA" in built_versions:
        print(f"  Running CUDA version...")
        
        start_time = time.time()
        success, _, stderr = run_command(
            f"../../build/KMEANS_cuda {dataset_path} 20 100 1.0 0.0001 "
            f"../../results/result_{dataset_name}_cuda_cluster.out 42"
        )
        end_time = time.time()
        
        if success:
            cuda_time = end_time - start_time
            speedup = baseline_time / cuda_time
            results['cuda'] = {'time': cuda_time, 'speedup': speedup, 'efficiency': speedup * 100}
            print(f"    CUDA: {cuda_time:.3f}s (speedup: {speedup:.2f}x)")
        else:
            print(f"    CUDA failed: {stderr}")
    
    return results

def generate_report(all_results, system_info):
    """Generate comprehensive performance report."""
    
    report_path = "../../logs/cluster_performance_report.txt"
    os.makedirs("../../logs", exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("K-means Cluster Performance Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        # System information
        f.write("System Information:\n")
        f.write(f"  SLURM Job ID: {system_info['job_id']}\n")
        f.write(f"  Node: {system_info['node']}\n")
        f.write(f"  CPU Model: {system_info['cpu_model']}\n")
        f.write(f"  Available CPUs: {system_info['cpus']}\n")
        f.write(f"  Memory: {system_info['memory']} MB\n")
        f.write(f"  Total Cores: {system_info['total_cores']}\n")
        f.write(f"  Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Performance results for each dataset
        for dataset_name, results in all_results.items():
            f.write(f"Dataset: {dataset_name}\n")
            f.write("-" * 30 + "\n")
            
            if 'sequential' in results:
                seq_time = results['sequential']['time']
                f.write(f"Sequential baseline: {seq_time:.3f}s\n\n")
                
                # OpenMP results
                if 'openmp' in results:
                    f.write("OpenMP Results:\n")
                    f.write("  Threads  Time(s)  Speedup  Efficiency(%)\n")
                    for threads, data in sorted(results['openmp'].items()):
                        f.write(f"  {threads:7d}  {data['time']:7.3f}  {data['speedup']:7.2f}  {data['efficiency']:11.1f}\n")
                    f.write("\n")
                
                # MPI results
                if 'mpi' in results:
                    f.write("MPI Results:\n")
                    f.write("  Procs    Time(s)  Speedup  Efficiency(%)\n")
                    for procs, data in sorted(results['mpi'].items()):
                        f.write(f"  {procs:5d}    {data['time']:7.3f}  {data['speedup']:7.2f}  {data['efficiency']:11.1f}\n")
                    f.write("\n")
                
                # CUDA results
                if 'cuda' in results:
                    cuda_data = results['cuda']
                    f.write("CUDA Results:\n")
                    f.write(f"  GPU: {cuda_data['time']:.3f}s (speedup: {cuda_data['speedup']:.2f}x)\n\n")
                
                # Best configuration
                best_speedup = 1.0
                best_config = "Sequential (1 core)"
                
                for version, data in results.items():
                    if version == 'sequential':
                        continue
                    elif version in ['openmp', 'mpi']:
                        for config, metrics in data.items():
                            if metrics['speedup'] > best_speedup:
                                best_speedup = metrics['speedup']
                                best_config = f"{version.upper()} ({config} {'threads' if version == 'openmp' else 'processes'})"
                    elif version == 'cuda':
                        if data['speedup'] > best_speedup:
                            best_speedup = data['speedup']
                            best_config = "CUDA (GPU)"
                
                f.write(f"Best configuration: {best_config} with {best_speedup:.2f}x speedup\n")
            
            f.write("\n" + "=" * 50 + "\n\n")
    
    print(f"Comprehensive report saved to: {report_path}")

def main():
    """Main function for cluster performance analysis."""
    print("K-means Cluster Performance Analysis")
    print("=" * 40)
    
    # Create necessary directories
    os.makedirs("../../logs", exist_ok=True)
    os.makedirs("../../results", exist_ok=True)
    
    # Get system information
    system_info = get_system_info()
    print(f"Running on SLURM job {system_info['job_id']} on node {system_info['node']}")
    print(f"Available CPUs: {system_info['cpus']}")
    
    # Build applications
    built_versions = build_applications()
    if not built_versions:
        print("ERROR: No versions built successfully")
        sys.exit(1)
    
    print(f"Built versions: {', '.join(built_versions)}")
    
    # Test datasets
    datasets = [
        ("../../data/input100D.inp", "100D"),
        ("../../data/input20D.inp", "20D"),
        ("../../data/input10D.inp", "10D"),
        ("../../data/input2D.inp", "2D"),
    ]
    
    all_results = {}
    
    for dataset_path, dataset_name in datasets:
        if os.path.exists(dataset_path):
            print(f"\nTesting dataset: {dataset_name}")
            results = run_performance_test(dataset_path, dataset_name, built_versions)
            if results:
                all_results[dataset_name] = results
        else:
            print(f"Dataset {dataset_path} not found, skipping...")
    
    if all_results:
        generate_report(all_results, system_info)
        print("\nCluster performance analysis completed successfully!")
    else:
        print("ERROR: No successful tests completed")
        sys.exit(1)

if __name__ == "__main__":
    main()
