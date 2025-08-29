#!/usr/bin/env python3
"""
Dataset Generator for K-means Performance Testing

This script analyzes the largest existing dataset (input100D2.inp) to determine
the data range, then generates larger synthetic datasets with the same characteristics.

Generated datasets:
- 200k_100.inp: 200,000 points, 100 dimensions
- 400k_100.inp: 400,000 points, 100 dimensions  
- 800k_100.inp: 800,000 points, 100 dimensions

Author: K-means Performance Analysis
Date: August 2025
"""

import numpy as np
import sys
import os

def analyze_dataset(filename):
    """
    Analyze the input dataset to find min/max values across all dimensions.
    
    Args:
        filename (str): Path to the input dataset file
        
    Returns:
        tuple: (min_value, max_value, num_points, num_dimensions)
    """
    print(f"Analyzing dataset: {filename}")
    
    # Read the dataset
    try:
        data = np.loadtxt(filename, delimiter='\t')
        print(f"Successfully loaded {data.shape[0]} points with {data.shape[1]} dimensions")
        
        # Find global min and max
        global_min = np.min(data)
        global_max = np.max(data)
        
        print(f"Data range: [{global_min}, {global_max}]")
        print(f"Data statistics:")
        print(f"  Mean: {np.mean(data):.2f}")
        print(f"  Std:  {np.std(data):.2f}")
        
        return global_min, global_max, data.shape[0], data.shape[1]
        
    except Exception as e:
        print(f"Error reading dataset: {e}")
        sys.exit(1)

def generate_dataset(num_points, num_dimensions, min_val, max_val, output_filename):
    """
    Generate a synthetic dataset with uniform random distribution.
    
    Args:
        num_points (int): Number of points to generate
        num_dimensions (int): Number of dimensions per point
        min_val (float): Minimum value for uniform distribution
        max_val (float): Maximum value for uniform distribution
        output_filename (str): Path to save the generated dataset
    """
    print(f"Generating {num_points:,} points with {num_dimensions} dimensions...")
    
    # Generate random data with uniform distribution
    np.random.seed(42)  # For reproducibility
    data = np.random.uniform(min_val, max_val, size=(num_points, num_dimensions))
    
    # Round to integers to match the original data format
    data = np.round(data).astype(int)
    
    # Save to file with tab-separated format
    print(f"Saving to: {output_filename}")
    np.savetxt(output_filename, data, delimiter='\t', fmt='%d')
    
    print(f"✓ Generated {output_filename} ({num_points:,} points)")

def main():
    """Main function to analyze existing data and generate new datasets."""
    
    # Configuration - use relative paths
    data_dir = "data"
    input_file = os.path.join(data_dir, "input100D2.inp")
    
    # Datasets to generate
    datasets_to_generate = [
        (200000, "200k_100.inp"),
        (400000, "400k_100.inp"), 
        (800000, "800k_100.inp")
    ]
    
    print("=" * 60)
    print("K-means Dataset Generator")
    print("=" * 60)
    
    # Step 1: Analyze the reference dataset
    min_val, max_val, ref_points, ref_dims = analyze_dataset(input_file)
    
    print(f"\nReference dataset characteristics:")
    print(f"  File: {input_file}")
    print(f"  Points: {ref_points:,}")
    print(f"  Dimensions: {ref_dims}")
    print(f"  Value range: [{min_val}, {max_val}]")
    
    # Step 2: Generate new datasets
    print("\n" + "=" * 60)
    print("Generating new datasets...")
    print("=" * 60)
    
    for num_points, filename in datasets_to_generate:
        output_path = os.path.join(data_dir, filename)
        generate_dataset(num_points, ref_dims, min_val, max_val, output_path)
        
        # Verify the generated file
        try:
            test_data = np.loadtxt(output_path, delimiter='\t', max_rows=1)
            print(f"  ✓ Verification: {len(test_data)} dimensions confirmed")
        except Exception as e:
            print(f"  ✗ Verification failed: {e}")
        
        print()
    
    print("=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)
    
    # Show final summary
    print("\nGenerated datasets summary:")
    for num_points, filename in datasets_to_generate:
        output_path = os.path.join(data_dir, filename)
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"  {filename:15} - {num_points:>7,} points - {file_size_mb:>6.1f} MB")
    
    print(f"\nAll datasets use:")
    print(f"  - {ref_dims} dimensions")
    print(f"  - Value range: [{min_val}, {max_val}]")
    print(f"  - Tab-separated integer format")
    print(f"  - Random seed: 42 (reproducible)")

if __name__ == "__main__":
    main()
