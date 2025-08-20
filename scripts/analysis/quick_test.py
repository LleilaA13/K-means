#!/usr/bin/env python3

import subprocess
import os

def quick_test():
    """Quick test of both sequential and parallel versions."""
    
    print("=== Quick Performance Test ===")
    
    # Change to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Test sequential version
    if os.path.exists("../../build/KMEANS_seq"):
        print("Testing sequential version...")
        try:
            result = subprocess.run([
                "./../../build/KMEANS_seq",
                "../../data/input100D.inp",
                "4", "10", "1.0", "0.001",  # Reduced iterations for quick test
                "../../results/test_seq.out",
                "42"
            ], capture_output=True, text=True, timeout=60)
            print(f"✅ Sequential: Exit code {result.returncode}")
        except Exception as e:
            print(f"❌ Sequential failed: {e}")
    else:
        print("❌ Sequential executable not found")
    
    # Test parallel version  
    if os.path.exists("../../build/KMEANS_omp"):
        print("Testing parallel version...")
        try:
            result = subprocess.run([
                "./../../build/KMEANS_omp", 
                "../../data/input100D.inp",
                "4", "10", "1.0", "0.001",  # Reduced iterations for quick test
                "../../results/test_omp.out",
                "42", "4"
            ], capture_output=True, text=True, timeout=60)
            print(f"✅ Parallel: Exit code {result.returncode}")
        except Exception as e:
            print(f"❌ Parallel failed: {e}")
    else:
        print("❌ Parallel executable not found")
    
    # Check log files
    if os.path.exists("../../logs/timing_log_seq.txt"):
        print("✅ Sequential log created")
    if os.path.exists("../../logs/timing_log_omp.txt"):
        print("✅ Parallel log created")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    quick_test()
