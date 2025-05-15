import numpy as np
import subprocess
import matplotlib.pyplot as plt

# Clear timing log first
with open("timing_log.txt", "w"):
    pass

# Run K-means 20 times with increasing seed
print("Running K-means with OpenMP...")
for i in range(1, 51):
    print(f"■", end="")
    subprocess.run(["./kmeans_omp", "test_files/input100D2.inp", "8", "100", "2.0", "0.001", "result_omp", str(i)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print()

times = np.loadtxt("timing_log.txt")
print(f"Runs:    {len(times)}")
print(f"Average: {np.mean(times):.4f} s")
print(f"Std Dev: {np.std(times):.4f} s")
print(f"Min:     {np.min(times):.4f} s")
print(f"Max:     {np.max(times):.4f} s")

# Plot it
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(times) + 1), times, marker='o', linestyle='-', color='dodgerblue')
plt.title("K-means Runtime Over Multiple Runs")
plt.xlabel("Run #")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.show()

