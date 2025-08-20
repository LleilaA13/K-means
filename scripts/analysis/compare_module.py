#!/usr/bin/env python3

import sys
import os
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def load_labels(filename):
    """Load cluster labels from file."""
    try:
        with open(filename, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"ERROR: File '{filename}' not found.")
        return None
    except ValueError as e:
        print(f"ERROR: Invalid data in '{filename}': {e}")
        return None

def compare_kmeans_outputs(seq_file, par_file):
    """Compare clustering outputs between sequential and parallel versions."""
    seq_labels = load_labels(seq_file)
    par_labels = load_labels(par_file)

    if seq_labels is None or par_labels is None:
        return

    if len(seq_labels) != len(par_labels):
        print("ERROR: Number of labels doesn't match.")
        print(f"Sequential: {len(seq_labels)} labels")
        print(f"Parallel: {len(par_labels)} labels")
        return

    ari = adjusted_rand_score(seq_labels, par_labels)
    nmi = normalized_mutual_info_score(seq_labels, par_labels)

    print(f"Clustering Comparison Results:")
    print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  Normalized Mutual Information (NMI): {nmi:.4f}")
    
    if ari > 0.9:
        print("  Status: Excellent agreement")
    elif ari > 0.7:
        print("  Status: Good agreement") 
    elif ari > 0.5:
        print("  Status: Moderate agreement")
    else:
        print("  Status: Poor agreement")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Command line usage
        compare_kmeans_outputs(sys.argv[1], sys.argv[2])
    else:
        # Default usage with new directory structure
        seq_file = "../../results/result_seq.out"
        omp_file = "../../results/result_omp.out"
        
        if os.path.exists(seq_file) and os.path.exists(omp_file):
            compare_kmeans_outputs(seq_file, omp_file)
        else:
            print("Usage: python compare_module.py <file1> <file2>")
            print("Or place result files in ../../results/ directory as:")
            print("  - ../../results/result_seq.out")
            print("  - ../../results/result_omp.out")
