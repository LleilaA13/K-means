# CUDA Parallelization Analysis for K-means Algorithm

## Overview

This document provides a comprehensive analysis of CUDA parallelization strategies for the K-means clustering algorithm, specifically optimized for the Quadro RTX 6000 GPUs available on the Sapienza University HPC cluster.

## Hardware Specifications

### Cluster GPU Resources

- **GPU Model**: NVIDIA Quadro RTX 6000
- **Architecture**: Turing (sm_75)
- **CUDA Cores per GPU**: 4,608
- **Streaming Multiprocessors (SMs)**: 72
- **CUDA Cores per SM**: 64
- **Memory**: 24 GB GDDR6
- **Memory Bandwidth**: 576 GB/s
- **Total GPUs Available**: 4 (across nodes 110, 111, 122)

### Per Job Allocation

When requesting `--gres=gpu:1`:

- **4,608 CUDA cores** (1 complete GPU)
- **24 GB GDDR6 memory**
- **576 GB/s memory bandwidth**
- **72 Streaming Multiprocessors**

## Optimal Grid and Block Configurations

### Point Assignment Kernel

```cuda
// Configuration for point-to-centroid assignment
dim3 blockSize(256);  // 256 threads per block (8 warps)
dim3 gridSize((numPoints + blockSize.x - 1) / blockSize.x);

// Alternative for 2D workload:
dim3 blockSize(128, 1);  // 128 threads for coalesced memory access
dim3 gridSize((numPoints + 127) / 128, K);  // Points × Centroids
```

### Centroid Update Kernel

```cuda
// Configuration for reduction-heavy operations
dim3 blockSize(512);  // Larger blocks for better reduction efficiency
dim3 gridSize(K);     // One block per centroid

// Shared memory requirement:
size_t sharedMem = blockSize.x * dimensions * sizeof(float);
```

### Memory Hierarchy Utilization

- **Shared Memory**: 64KB per SM → Cache frequently accessed centroids
- **Registers**: 65,536 per SM → Minimize register pressure
- **Global Memory**: Optimize for coalesced access patterns

## CUDA Kernel Implementation

### Constant Memory Optimization

```cuda
// Declare constant memory for frequently accessed parameters
__constant__ int c_numPoints;    // Number of data points
__constant__ int c_dimensions;   // Number of dimensions per point
__constant__ int c_K;           // Number of clusters

// Host function to set constant memory
void setConstantMemory(int numPoints, int dimensions, int K) {
    cudaMemcpyToSymbol(c_numPoints, &numPoints, sizeof(int));
    cudaMemcpyToSymbol(c_dimensions, &dimensions, sizeof(int));
    cudaMemcpyToSymbol(c_K, &K, sizeof(int));
}
```

### Kernel 1: Point Assignment (Optimized with Constants)

```cuda
__global__ void assignPointsToCentroids(
    float* points,           // [c_numPoints * c_dimensions]
    float* centroids,        // [c_K * c_dimensions]
    int* assignments,        // [c_numPoints]
    int* changes             // [1] - atomic counter for changes
) {
    extern __shared__ float sharedCentroids[];  // Cache centroids in shared memory

    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= c_numPoints) return;

    // Load centroids into shared memory (collaborative loading)
    for (int i = threadIdx.x; i < c_K * c_dimensions; i += blockDim.x) {
        sharedCentroids[i] = centroids[i];
    }
    __syncthreads();

    float minDistance = FLT_MAX;
    int bestCentroid = 0;
    int oldAssignment = assignments[pointIdx];

    // Calculate distance to each centroid
    for (int k = 0; k < c_K; k++) {
        float distance = 0.0f;

        // Manual loop unrolling can be done if needed for specific dimension counts
        for (int d = 0; d < c_dimensions; d++) {
            float diff = points[pointIdx * c_dimensions + d] -
                        sharedCentroids[k * c_dimensions + d];
            distance += diff * diff;
        }        if (distance < minDistance) {
            minDistance = distance;
            bestCentroid = k;
        }
    }

    assignments[pointIdx] = bestCentroid;

    // Count changes using atomic operation
    if (oldAssignment != bestCentroid) {
        atomicAdd(changes, 1);
    }
}
```

### Kernel 2: Centroid Update (Optimized with Constants)

````cuda
__global__ void updateCentroids(
    float* points,           // [c_numPoints * c_dimensions]
    int* assignments,        // [c_numPoints]
    float* newCentroids,     // [c_K * c_dimensions]
    int* pointCounts         // [c_K]
) {
    extern __shared__ float sharedData[];

    int centroidIdx = blockIdx.x;  // One block per centroid
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (centroidIdx >= c_K) return;

    // Initialize shared memory for reduction
    float* centroidSum = &sharedData[tid * c_dimensions];
    int pointCount = 0;

    // Initialize
    for (int d = 0; d < c_dimensions; d++) {
        centroidSum[d] = 0.0f;
    }

    // Accumulate points assigned to this centroid
    for (int i = tid; i < c_numPoints; i += blockSize) {
        if (assignments[i] == centroidIdx) {
            pointCount++;
            for (int d = 0; d < c_dimensions; d++) {
                centroidSum[d] += points[i * c_dimensions + d];
            }
        }
    }

    // Reduction within block using warp shuffles
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            pointCount += __shfl_down_sync(0xFFFFFFFF, pointCount, stride);
            for (int d = 0; d < c_dimensions; d++) {
                centroidSum[d] += __shfl_down_sync(0xFFFFFFFF, centroidSum[d], stride);
            }
        }
    }

    // Write result
    if (tid == 0) {
        pointCounts[centroidIdx] = pointCount;
        for (int d = 0; d < c_dimensions; d++) {
            newCentroids[centroidIdx * c_dimensions + d] =
                (pointCount > 0) ? centroidSum[d] / pointCount : 0.0f;
        }
    }
}

## Simplified Kernel Approach (Much Easier!)

### Simple Kernel 2: Atomic Operations Approach

```cuda
__global__ void updateCentroids_simple(
    float* points,           // [c_numPoints * c_dimensions]
    int* assignments,        // [c_numPoints]
    float* newCentroids,     // [c_K * c_dimensions]
    int* pointCounts         // [c_K]
) {
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pointIdx >= c_numPoints) return;

    int centroidIdx = assignments[pointIdx] - 1;  // Convert 1-based to 0-based

    if (centroidIdx >= 0 && centroidIdx < c_K) {
        // Atomically increment point count for this centroid
        atomicAdd(&pointCounts[centroidIdx], 1);

        // Atomically add this point's coordinates to the centroid sum
        for (int d = 0; d < c_dimensions; d++) {
            atomicAdd(&newCentroids[centroidIdx * c_dimensions + d],
                     points[pointIdx * c_dimensions + d]);
        }
    }
}

// Simple kernel to divide by counts (finalize centroids)
__global__ void finalizeCentroids_kernel(float* newCentroids, int* pointCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalElements = c_K * c_dimensions;

    if (idx >= totalElements) return;

    int centroidIdx = idx / c_dimensions;
    int count = pointCounts[centroidIdx];

    if (count > 0) {
        newCentroids[idx] /= count;
    }
}
```

### Even Simpler: One Thread Per Centroid

```cuda
__global__ void updateCentroids_onePerCentroid(
    float* points,           // [c_numPoints * c_dimensions]
    int* assignments,        // [c_numPoints]
    float* newCentroids,     // [c_K * c_dimensions]
    int* pointCounts         // [c_K]
) {
    int centroidIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (centroidIdx >= c_K) return;

    // Initialize this centroid
    int count = 0;
    for (int d = 0; d < c_dimensions; d++) {
        newCentroids[centroidIdx * c_dimensions + d] = 0.0f;
    }

    // Sum all points assigned to this centroid
    for (int i = 0; i < c_numPoints; i++) {
        if (assignments[i] - 1 == centroidIdx) {  // Convert 1-based to 0-based
            count++;
            for (int d = 0; d < c_dimensions; d++) {
                newCentroids[centroidIdx * c_dimensions + d] +=
                    points[i * c_dimensions + d];
            }
        }
    }

    // Calculate mean (no atomics needed!)
    pointCounts[centroidIdx] = count;
    if (count > 0) {
        for (int d = 0; d < c_dimensions; d++) {
            newCentroids[centroidIdx * c_dimensions + d] /= count;
        }
    }
}
```

**Benefits of Simple Approaches:**
- **Easy to understand and debug**
- **No complex reduction logic**
- **Straightforward implementation**
- **Still provides good GPU acceleration**

**Choose based on your dataset:**
- **Atomic approach**: Good for large datasets with many points per cluster
- **One-per-centroid**: Perfect for small-medium datasets and easy debugging

## Complete Algorithm Structure

### Main CUDA Implementation with Constant Memory

```cuda
void cudaKMeans(float* h_points, int* h_assignments, float* h_centroids,
                int numPoints, int dimensions, int K, int maxIterations) {

    // Set constant memory once at the beginning
    setConstantMemory(numPoints, dimensions, K);

    // Memory allocation
    float *d_points, *d_centroids, *d_newCentroids;
    int *d_assignments, *d_pointCounts, *d_changes;

    cudaMalloc(&d_points, numPoints * dimensions * sizeof(float));
    cudaMalloc(&d_centroids, K * dimensions * sizeof(float));
    cudaMalloc(&d_newCentroids, K * dimensions * sizeof(float));
    cudaMalloc(&d_assignments, numPoints * sizeof(int));
    cudaMalloc(&d_pointCounts, K * sizeof(int));
    cudaMalloc(&d_changes, sizeof(int));

    // Copy initial data
    cudaMemcpy(d_points, h_points, numPoints * dimensions * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, K * dimensions * sizeof(float),
               cudaMemcpyHostToDevice);

    // Kernel configuration (no need to pass numPoints, dimensions, K)
    dim3 assignBlock(256);
    dim3 assignGrid((numPoints + 255) / 256);

    dim3 updateBlock(512);
    dim3 updateGrid(K);

    // Shared memory calculations using constant values
    size_t sharedMemAssign = K * dimensions * sizeof(float);
    size_t sharedMemUpdate = updateBlock.x * dimensions * sizeof(float);

    for (int iter = 0; iter < maxIterations; iter++) {
        // Reset change counter
        int zero = 0;
        cudaMemcpy(d_changes, &zero, sizeof(int), cudaMemcpyHostToDevice);

        // Step 1: Assign points to centroids (simplified parameter list)
        assignPointsToCentroids<<<assignGrid, assignBlock, sharedMemAssign>>>(
            d_points, d_centroids, d_assignments, d_changes);

        // Step 2: Update centroids (simplified parameter list)
        updateCentroids<<<updateGrid, updateBlock, sharedMemUpdate>>>(
            d_points, d_assignments, d_newCentroids, d_pointCounts);

        // Swap centroids
        float* temp = d_centroids;
        d_centroids = d_newCentroids;
        d_newCentroids = temp;

        cudaDeviceSynchronize();

        // Check convergence
        int h_changes;
        cudaMemcpy(&h_changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_changes <= minChanges) {
            break; // Convergence reached
        }
    }

    // Copy results back
    cudaMemcpy(h_assignments, d_assignments, numPoints * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, K * dimensions * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_newCentroids);
    cudaFree(d_assignments);
    cudaFree(d_pointCounts);
    cudaFree(d_changes);
}
````

### Benefits of Constant Memory Optimization

1. **Faster Access**: Constant memory is cached and provides faster access than global memory parameters
2. **Reduced Parameter Passing**: Cleaner kernel signatures with fewer parameters
3. **Better Register Usage**: Frees up registers that would otherwise hold these values
4. **Single Assignment**: Values are set once and reused across all kernel invocations
5. **Hardware Optimization**: NVIDIA GPUs have dedicated constant memory cache

## Performance Analysis

### Theoretical Performance

With 4,608 CUDA cores at 1.77 GHz boost clock:

- **Peak Performance**: ~16.3 TFLOPS (single precision)
- **Memory Bandwidth**: 576 GB/s
- **Expected K-means Speedup**: 10-30x over sequential CPU

### Bottleneck Analysis

#### Memory Bound Scenarios

- **Characteristics**: Large datasets (>1M points), High dimensions (>100D)
- **Solutions**:
  - Coalesced memory access patterns
  - Shared memory caching for centroids
  - Minimize global memory transactions

#### Compute Bound Scenarios

- **Characteristics**: Many clusters (K>50), Small datasets (<10K points)
- **Solutions**:
  - Optimize arithmetic intensity
  - Use tensor cores if applicable
  - Minimize divergent branching

#### Synchronization Overhead

- **Characteristics**: Frequent CPU-GPU transfers
- **Solutions**:
  - Keep data on GPU between iterations
  - Implement convergence checking on GPU
  - Use CUDA streams for overlapped execution

### Occupancy Optimization

```cuda
// Check and optimize occupancy
int blockSize = 256;
int minGridSize, maxBlockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize,
                                   assignPointsToCentroids, 0, 0);

// For Quadro RTX 6000:
// - 72 SMs × 1024 max threads per SM = 73,728 max concurrent threads
// - With 256 thread blocks: 288 concurrent blocks
// - Target: 75-100% occupancy for optimal performance
```

### Memory Access Patterns

#### Optimal Coalesced Access

```cuda
// Data layout for coalesced access:
// Points: [point0_dim0, point0_dim1, ..., point1_dim0, point1_dim1, ...]

// Thread access pattern:
// Thread 0: points[0], points[1], points[2], ...
// Thread 1: points[dimensions], points[dimensions+1], ...
// Thread 2: points[2*dimensions], points[2*dimensions+1], ...
```

#### Shared Memory Usage

```cuda
// Centroid caching strategy:
extern __shared__ float sharedCentroids[];

// Load centroids collaboratively:
for (int i = threadIdx.x; i < K * dimensions; i += blockDim.x) {
    sharedCentroids[i] = centroids[i];
}
__syncthreads();
```

## Implementation Roadmap

### Phase 1: Basic Parallelization

1. **Point Assignment Kernel**

   - Parallel distance calculation
   - Basic memory management
   - Simple thread-to-point mapping

2. **Centroid Update**
   - Atomic operations for accumulation
   - Basic reduction for averaging
   - CPU-based convergence checking

### Phase 2: Optimization

1. **Memory Optimizations**

   - Shared memory for centroid caching
   - Coalesced memory access patterns
   - Minimize global memory transactions

2. **Compute Optimizations**
   - Reduction-based centroid updates
   - Warp-level primitives
   - Loop unrolling and vectorization

### Phase 3: Advanced Features

1. **Multi-GPU Support**

   - Data partitioning across GPUs
   - Inter-GPU communication
   - Load balancing strategies

2. **Streaming and Overlapping**

   - CUDA streams for concurrent execution
   - Overlapped computation and communication
   - Asynchronous memory transfers

3. **Dynamic Features**
   - GPU-based convergence checking
   - Dynamic parallelism for adaptive K
   - Memory pool management

## Expected Performance Gains

### Dataset Size vs Speedup

| Dataset Size | Points   | Expected Speedup | Limiting Factor         |
| ------------ | -------- | ---------------- | ----------------------- |
| Small        | 1K-10K   | 5-10x            | Kernel launch overhead  |
| Medium       | 10K-100K | 15-25x           | Balanced compute/memory |
| Large        | 100K-1M  | 20-35x           | Memory bandwidth        |
| Very Large   | >1M      | 25-40x           | Memory bandwidth        |

### Dimensionality Impact

| Dimensions | Memory per Point | Expected Performance           |
| ---------- | ---------------- | ------------------------------ |
| 2D-10D     | 8-40 bytes       | Compute bound, high speedup    |
| 20D-50D    | 80-200 bytes     | Balanced workload              |
| 100D+      | 400+ bytes       | Memory bound, moderate speedup |

### Cluster Count Impact

| K (Clusters) | Shared Memory Usage | Performance Notes               |
| ------------ | ------------------- | ------------------------------- |
| K ≤ 10       | <2KB per block      | Optimal shared memory usage     |
| 10 < K ≤ 50  | 2-10KB per block    | Good performance                |
| K > 50       | >10KB per block     | May exceed shared memory limits |

## Optimization Guidelines

### Memory Management

1. **Minimize Host-Device Transfers**

   - Keep data on GPU between iterations
   - Use pinned memory for faster transfers
   - Implement GPU-based convergence checking

2. **Optimize Memory Access**
   - Ensure coalesced access patterns
   - Use shared memory for frequently accessed data
   - Align data structures to memory boundaries

### Compute Optimization

1. **Thread Utilization**

   - Target 75-100% occupancy
   - Balance threads per block with shared memory usage
   - Use warp-level primitives for reductions

2. **Algorithmic Improvements**
   - Early termination for convergence
   - Approximate distance calculations when appropriate
   - Triangle inequality for distance pruning

### Debugging and Profiling

1. **Tools**

   - `nvidia-smi` for GPU utilization monitoring
   - `nvprof` or `nsight` for detailed profiling
   - CUDA error checking macros

2. **Metrics to Monitor**
   - Kernel execution time
   - Memory throughput
   - SM utilization
   - Occupancy percentage

## Cluster-Specific Considerations

### SLURM Configuration

```bash
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --partition=students      # Use students partition
#SBATCH --cpus-per-task=4         # 4 CPU cores for GPU support
#SBATCH --mem=8GB                 # 8GB RAM
```

### Compilation Flags

```bash
# Optimized compilation for Quadro RTX 6000
nvcc -O3 -arch=sm_75 \
     -use_fast_math \
     -Xptxas -O3 \
     src/KMEANS_cuda.cu -lm -o build/KMEANS_cuda
```

### Runtime Environment

```bash
# Optimal GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Conclusion

The CUDA parallelization of K-means on the Quadro RTX 6000 can achieve significant performance improvements through:

1. **Efficient kernel design** utilizing 4,608 CUDA cores
2. **Optimized memory access patterns** leveraging 576 GB/s bandwidth
3. **Proper resource utilization** across 72 streaming multiprocessors
4. **Algorithm-specific optimizations** for clustering workloads

Expected overall speedup: **10-30x** over sequential CPU implementation, with actual performance depending on dataset characteristics and implementation quality.

The provided kernel implementations and optimization strategies form a solid foundation for achieving optimal performance on this specific hardware configuration.
