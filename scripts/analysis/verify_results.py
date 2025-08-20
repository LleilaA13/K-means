#!/usr/bin/env python3
"""
Dynamic K-means Results Verification Tool
Automatically finds and compares all parallel results with sequential baseline
Uses the existing compare_module.py for comparisons
"""

import os
import sys
import glob
import subprocess
from pathlib import Path

def find_sequential_baseline(results_dir):
    """Find the sequential baseline file in results directory."""
    # Look for files containing 'seq' in the name
    seq_patterns = [
        "*_seq_*.out",
        "*_seq.out", 
        "result_seq*.out"
    ]
    
    for pattern in seq_patterns:
        seq_files = glob.glob(os.path.join(results_dir, pattern))
        if seq_files:
            return seq_files[0]  # Return first found
    
    return None

def find_parallel_results(results_dir):
    """Find all parallel result files, excluding sequential ones."""
    # Get all .out files
    all_files = glob.glob(os.path.join(results_dir, "*.out"))
    
    parallel_files = []
    
    for file in all_files:
        filename = os.path.basename(file)
        
        # Skip sequential files
        if '_seq_' in filename or '_seq.' in filename:
            continue
            
        # Include files that look like parallel results
        parallel_indicators = ['_omp_', '_mpi_', '_cuda_', 'threads', 'procs']
        
        if any(indicator in filename.lower() for indicator in parallel_indicators):
            parallel_files.append(file)
        elif filename.startswith('result_') and filename.endswith('.out'):
            # Include other result files that might be parallel
            parallel_files.append(file)
    
    return sorted(parallel_files)

def run_comparison(seq_file, par_file):
    """Run comparison using existing compare_module.py."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compare_script = os.path.join(script_dir, 'compare_module.py')
    
    if not os.path.exists(compare_script):
        print(f"ERROR: compare_module.py not found at {compare_script}")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, compare_script, seq_file, par_file
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
            return True
        else:
            print(f"ERROR: {result.stderr.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print("ERROR: Comparison timed out")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Main function."""
    # Get results directory
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Default to ../../results relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, '..', '..', 'results')
    
    results_dir = os.path.abspath(results_dir)
    
    print("=" * 60)
    print("K-means Results Verification Tool")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Find sequential baseline
    seq_file = find_sequential_baseline(results_dir)
    if not seq_file:
        print("ERROR: No sequential baseline file found!")
        print("Looking for files matching: *_seq_*.out, *_seq.out, result_seq*.out")
        sys.exit(1)
    
    print(f"Sequential baseline: {os.path.basename(seq_file)}")
    
    # Find parallel results
    parallel_files = find_parallel_results(results_dir)
    if not parallel_files:
        print("No parallel result files found!")
        sys.exit(1)
    
    print(f"Found {len(parallel_files)} parallel result files")
    print()
    
    # Compare each parallel file with sequential
    total_comparisons = 0
    successful_comparisons = 0
    
    for par_file in parallel_files:
        filename = os.path.basename(par_file)
        print(f"Comparing: {filename}")
        print("-" * 50)
        
        total_comparisons += 1
        if run_comparison(seq_file, par_file):
            successful_comparisons += 1
        
        print()  # Empty line for readability
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total comparisons: {total_comparisons}")
    print(f"Successful comparisons: {successful_comparisons}")
    
    if successful_comparisons == total_comparisons:
        print("🎉 ALL RESULTS CORRECT!")
        print("All parallel implementations produce identical clustering!")
    else:
        failed = total_comparisons - successful_comparisons
        print(f"⚠️  {failed} comparisons failed")
        print("Check the results above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
