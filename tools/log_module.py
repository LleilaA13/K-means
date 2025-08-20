import numpy as np
import subprocess
import matplotlib.pyplot as plt

log_file = "timing_log_omp.txt"

# Clear timing log first
with open(log_file, "w"):
    pass

# Run K-means 6 times with increasing seed
print("Running K-means... ")
for i in range(1, 7):
    print(f"■■", end="")
    # Reminder: Modify the command below to match your environment
    subprocess.run(["./kmeans_omp", "test_files/input100D2.inp", "20", "100", "1.0", "0.0001", "result_omp", "1", str(2**i)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print()

times = np.loadtxt(log_file)
print(f"Runs:    {len(times)}")
print(f"Average: {np.mean(times):.4f} s")
print(f"Std Dev: {np.std(times):.4f} s")
print(f"Min:     {np.min(times):.4f} s")
print(f"Max:     {np.max(times):.4f} s")

print(59.60/times)
# Plot it
"""
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(times) + 1), times, marker='o', linestyle='-', color='dodgerblue')
plt.title("K-means Runtime Over Multiple Runs")
plt.xlabel("Run #")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.show()
"""
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(times) + 1), 59.60/times, marker='o', linestyle='-', color='dodgerblue')
plt.title("K-means Runtime Over Multiple Runs")
plt.xlabel("2^# - threads")
plt.ylabel("Speed-up  ")
plt.grid(True)
plt.tight_layout()
plt.show()