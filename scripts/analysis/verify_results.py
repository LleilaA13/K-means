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

def extract_dataset_name(filename):
    """Extract dataset name from result filename."""
    # Remove .out extension
    name = filename.replace('.out', '')
    
    # Common patterns for result files:
    # seq_dataset_runX
    # omp_dataset_Xt_runX
    # mpi_dataset_Xp_runX
    # mpi_omp_dataset_Xp_Yt_runX
    # cuda_dataset_runX
    # result_dataset_implementation
    
    # Split by underscores
    parts = name.split('_')
    
    if len(parts) < 2:
        return None
    
    # Handle different naming patterns
    if parts[0] in ['seq', 'omp', 'mpi', 'cuda']:
        # Format: implementation_dataset_config_runX
        if len(parts) >= 2:
            return parts[1]  # dataset is second part
    elif parts[0] == 'result':
        # Format: result_dataset_implementation or result_dataset_config
        if len(parts) >= 2:
            return parts[1]  # dataset is second part
    
    # If no clear pattern, try to find dataset name
    # Look for common dataset patterns
    dataset_indicators = ['input', 'data', '100D', '200k', '400k', '800k']
    for part in parts:
        if any(indicator in part for indicator in dataset_indicators):
            return part
    
    return None

def find_sequential_baselines(results_dir):
    """Find all sequential baseline files grouped by dataset."""
    seq_patterns = [
        "*_seq_*.out",
        "*_seq.out", 
        "result_seq*.out"
    ]
    
    seq_files = []
    for pattern in seq_patterns:
        seq_files.extend(glob.glob(os.path.join(results_dir, pattern)))
    
    # Group by dataset
    seq_by_dataset = {}
    for seq_file in seq_files:
        filename = os.path.basename(seq_file)
        dataset = extract_dataset_name(filename)
        if dataset:
            seq_by_dataset[dataset] = seq_file
    
    return seq_by_dataset

def find_parallel_results(results_dir):
    """Find all parallel result files grouped by dataset."""
    # Get all .out files
    all_files = glob.glob(os.path.join(results_dir, "*.out"))
    
    parallel_by_dataset = {}
    
    for file in all_files:
        filename = os.path.basename(file)
        
        # Skip sequential files
        if '_seq_' in filename or '_seq.' in filename:
            continue
            
        # Include files that look like parallel results
        parallel_indicators = ['_omp_', '_mpi_', '_cuda_', 'threads', 'procs']
        
        is_parallel = False
        if any(indicator in filename.lower() for indicator in parallel_indicators):
            is_parallel = True
        elif filename.startswith('result_') and filename.endswith('.out'):
            # Include other result files that might be parallel
            is_parallel = True
        
        if is_parallel:
            dataset = extract_dataset_name(filename)
            if dataset:
                if dataset not in parallel_by_dataset:
                    parallel_by_dataset[dataset] = []
                parallel_by_dataset[dataset].append(file)
    
    # Sort files within each dataset
    for dataset in parallel_by_dataset:
        parallel_by_dataset[dataset].sort()
    
    return parallel_by_dataset

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
    
    # Find sequential baselines grouped by dataset
    seq_by_dataset = find_sequential_baselines(results_dir)
    if not seq_by_dataset:
        print("ERROR: No sequential baseline files found!")
        print("Looking for files matching: *_seq_*.out, *_seq.out, result_seq*.out")
        sys.exit(1)
    
    print(f"Found sequential baselines for datasets: {list(seq_by_dataset.keys())}")
    
    # Find parallel results grouped by dataset
    parallel_by_dataset = find_parallel_results(results_dir)
    if not parallel_by_dataset:
        print("No parallel result files found!")
        sys.exit(1)
    
    print(f"Found parallel results for datasets: {list(parallel_by_dataset.keys())}")
    print()
    
    # Compare each parallel file with corresponding sequential baseline
    total_comparisons = 0
    successful_comparisons = 0
    datasets_processed = 0
    
    for dataset in sorted(set(seq_by_dataset.keys()) & set(parallel_by_dataset.keys())):
        seq_file = seq_by_dataset[dataset]
        parallel_files = parallel_by_dataset[dataset]
        
        print(f"📊 DATASET: {dataset}")
        print("=" * 50)
        print(f"Sequential baseline: {os.path.basename(seq_file)}")
        print(f"Parallel implementations: {len(parallel_files)}")
        print()
        
        datasets_processed += 1
        
        for par_file in parallel_files:
            filename = os.path.basename(par_file)
            print(f"Comparing: {filename}")
            print("-" * 40)
            
            total_comparisons += 1
            if run_comparison(seq_file, par_file):
                successful_comparisons += 1
                print("✅ MATCH")
            else:
                print("❌ MISMATCH")
            
            print()  # Empty line for readability
        
        print()  # Extra line between datasets
    
    # Report on datasets without both sequential and parallel results
    seq_only = set(seq_by_dataset.keys()) - set(parallel_by_dataset.keys())
    parallel_only = set(parallel_by_dataset.keys()) - set(seq_by_dataset.keys())
    
    if seq_only:
        print("⚠️  Datasets with sequential results only:")
        for dataset in sorted(seq_only):
            print(f"   - {dataset}")
        print()
    
    if parallel_only:
        print("⚠️  Datasets with parallel results only (no sequential baseline):")
        for dataset in sorted(parallel_only):
            print(f"   - {dataset}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Datasets processed: {datasets_processed}")
    print(f"Total comparisons: {total_comparisons}")
    print(f"Successful comparisons: {successful_comparisons}")
    
    if successful_comparisons == total_comparisons and total_comparisons > 0:
        print("🎉 ALL RESULTS CORRECT!")
        print("All parallel implementations produce identical clustering!")
    elif total_comparisons == 0:
        print("⚠️  No comparisons could be performed")
        print("Check that you have both sequential and parallel results for the same datasets")
        sys.exit(1)
    else:
        failed = total_comparisons - successful_comparisons
        print(f"⚠️  {failed} comparisons failed")
        print("Check the results above for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
