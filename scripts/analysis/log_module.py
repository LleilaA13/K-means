#!/usr/bin/env python3

import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
import sys

def run_performance_test():
    """Run K-means performance tests with sequential and parallel versions."""
    
    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Ensure directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Check if executables exist
    seq_executable = "../../build/KMEANS_seq"
    omp_executable = "../../build/KMEANS_omp"
    
    if not os.path.exists(seq_executable):
        print(f"WARNING: {seq_executable} not found. Run 'make KMEANS_seq' to build it.")
        seq_executable = None
    
    if not os.path.exists(omp_executable):
        print(f"ERROR: {omp_executable} not found. Please run 'make KMEANS_omp' first.")
        return None
    
    # Check if input file exists
    input_file = "../../data/input100D.inp"
    if not os.path.exists(input_file):
        print(f"ERROR: {input_file} not found.")
        return None
    
    results = {}
    
    # Run sequential version first (baseline)
    if seq_executable:
        print("Running sequential K-means...")
        seq_log = "../../logs/timing_log_seq.txt"
        with open(seq_log, "w"):
            pass
        
        try:
            print("■■", end="", flush=True)
            subprocess.run([
                f"./{seq_executable}",
                input_file,
                "20",           # clusters
                "100",          # max iterations
                "1.0",          # min changes %
                "0.0001",       # threshold
                "../../results/result_seq.out",
                "1"             # seed
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
               check=True, timeout=600)  # 10 min timeout for sequential
            
            seq_times = np.loadtxt(seq_log)
            # Handle both scalar and array cases
            if np.isscalar(seq_times):
                seq_time = float(seq_times)
            elif seq_times.ndim == 0:  # 0-dimensional array
                seq_time = float(seq_times.item())
            else:  # 1D array
                seq_time = float(seq_times[0])
            results['sequential'] = seq_time
            print(f" Sequential: {seq_time:.4f}s")
        except Exception as e:
            print(f"\nWARNING: Sequential run failed: {e}")
            results['sequential'] = None
    
    # Run parallel version with different thread counts
    print("Running parallel K-means...")
    omp_log = "../../logs/timing_log_omp.txt"
    with open(omp_log, "w"):
        pass
    
    thread_counts = []
    for i in range(1, 7):
        threads = 2**i
        thread_counts.append(threads)
        print(f"■■", end="", flush=True)
        
        try:
            subprocess.run([
                f"./{omp_executable}",
                input_file,
                "20",           # clusters
                "100",          # max iterations
                "1.0",          # min changes %
                "0.0001",       # threshold
                f"../../results/result_omp_{threads}threads.out",
                "1",            # seed
                str(threads)    # thread count
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
               check=True, timeout=300)  # 5 min timeout
        except subprocess.TimeoutExpired:
            print(f"\nERROR: Timeout with {threads} threads")
            continue
        except subprocess.CalledProcessError as e:
            print(f"\nERROR: Failed with {threads} threads (exit code: {e.returncode})")
            continue
    
    print()
    
    # Load parallel results
    if not os.path.exists(omp_log):
        print(f"ERROR: Log file {omp_log} not found")
        return None
        
    try:
        omp_times = np.loadtxt(omp_log)
        if len(omp_times) == 0:
            print("ERROR: No parallel timing data found")
            return None
        
        results['parallel_times'] = omp_times
        results['thread_counts'] = thread_counts[:len(omp_times)]
        return results
        
    except Exception as e:
        print(f"ERROR loading parallel timing data: {e}")
        return None

def analyze_results(results):
    """Analyze and display performance results."""
    
    seq_time = results.get('sequential')
    parallel_times = results.get('parallel_times')
    thread_counts = results.get('thread_counts')
    
    if parallel_times is None:
        print("ERROR: No parallel timing data to analyze")
        return None
    
    print(f"\n=== Performance Analysis ===")
    
    # Sequential results
    if seq_time is not None:
        print(f"Sequential version: {seq_time:.4f}s")
        baseline = seq_time
    else:
        print("Sequential version: Not available")
        # Use single-threaded parallel as baseline
        baseline = parallel_times[0] if len(parallel_times) > 0 else 1.0
        print(f"Using single-threaded parallel as baseline: {baseline:.4f}s")
    
    # Parallel results
    print(f"\nParallel version results:")
    print(f"  Runs:    {len(parallel_times)}")
    print(f"  Average: {np.mean(parallel_times):.4f}s")
    print(f"  Std Dev: {np.std(parallel_times):.4f}s")
    print(f"  Min:     {np.min(parallel_times):.4f}s")
    print(f"  Max:     {np.max(parallel_times):.4f}s")
    
    # Calculate speedups
    speedups = baseline / parallel_times
    efficiency = speedups / np.array(thread_counts) * 100  # Parallel efficiency
    
    print(f"\n=== Speedup Analysis ===")
    print(f"{'Threads':<8} {'Time (s)':<10} {'Speedup':<10} {'Efficiency':<12}")
    print("-" * 42)
    
    if seq_time is not None:
        print(f"{'1 (seq)':<8} {seq_time:<10.4f} {'1.00x':<10} {'100.0%':<12}")
    
    for i, (threads, time, speedup, eff) in enumerate(zip(thread_counts, parallel_times, speedups, efficiency)):
        print(f"{threads:<8} {time:<10.4f} {speedup:<10.2f}x {eff:<12.1f}%")
    
    # Find optimal configuration
    best_speedup_idx = np.argmax(speedups)
    best_threads = thread_counts[best_speedup_idx]
    best_speedup = speedups[best_speedup_idx]
    
    print(f"\n=== Optimization Summary ===")
    print(f"Best configuration: {best_threads} threads")
    print(f"Best speedup: {best_speedup:.2f}x")
    print(f"Best time: {parallel_times[best_speedup_idx]:.4f}s")
    
    if seq_time is not None:
        total_speedup = seq_time / parallel_times[best_speedup_idx]
        print(f"Total speedup vs sequential: {total_speedup:.2f}x")
    
    return {
        'baseline': baseline,
        'parallel_times': parallel_times,
        'thread_counts': thread_counts,
        'speedups': speedups,
        'efficiency': efficiency,
        'sequential_time': seq_time
    }

def plot_results(analysis_data):
    """Create comprehensive performance visualization plots."""
    
    baseline = analysis_data['baseline']
    parallel_times = analysis_data['parallel_times']
    thread_counts = analysis_data['thread_counts']
    speedups = analysis_data['speedups']
    efficiency = analysis_data['efficiency']
    seq_time = analysis_data['sequential_time']
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Execution Time Comparison
    ax1.plot(thread_counts, parallel_times, marker='o', linestyle='-', 
             color='dodgerblue', linewidth=2, markersize=8, label='Parallel')
    
    if seq_time is not None:
        ax1.axhline(y=seq_time, color='red', linestyle='--', linewidth=2, 
                   label=f'Sequential ({seq_time:.3f}s)')
    
    ax1.set_title("Execution Time Comparison", fontsize=14)
    ax1.set_xlabel("Number of Threads", fontsize=12)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.legend()
    
    # Plot 2: Speedup
    ax2.plot(thread_counts, speedups, marker='o', linestyle='-', 
             color='green', linewidth=2, markersize=8, label='Actual Speedup')
    
    # Ideal speedup line
    ideal_speedup = np.array(thread_counts) / thread_counts[0]
    ax2.plot(thread_counts, ideal_speedup, '--', color='gray', 
             alpha=0.7, label='Ideal Speedup')
    
    ax2.set_title("Speedup Analysis", fontsize=14) 
    ax2.set_xlabel("Number of Threads", fontsize=12)
    ax2.set_ylabel("Speedup Factor", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    ax2.legend()
    
    # Plot 3: Parallel Efficiency
    ax3.plot(thread_counts, efficiency, marker='s', linestyle='-', 
             color='orange', linewidth=2, markersize=8)
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='Perfect Efficiency')
    ax3.set_title("Parallel Efficiency", fontsize=14)
    ax3.set_xlabel("Number of Threads", fontsize=12)
    ax3.set_ylabel("Efficiency (%)", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_ylim(0, 120)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_file = "../../results/performance_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {plot_file}")
    
    plt.show()
    
    return plot_file

if __name__ == "__main__":
    print("=== K-means Performance Analysis Tool ===")
    print("This script will run both sequential and parallel versions")
    print("and provide comprehensive performance analysis.\n")
    
    results = run_performance_test()
    if results is not None:
        analysis_data = analyze_results(results)
        if analysis_data is not None:
            plot_results(analysis_data)
            print("\n=== Analysis Complete ===")
        else:
            print("Analysis failed.")
            sys.exit(1)
    else:
        print("Performance test failed.")
        sys.exit(1)