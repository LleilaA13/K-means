/*
 * K-Means Clustering Algorithm - CUDA Implementation
 *
 * High-performance GPU-accelerated clustering with advanced optimizations:
 * - Warp-level reduction for maximum distance finding
 * - Shared memory optimization for centroid access
 * - Dynamic block sizing based on device capabilities
 * - Async memory operations with CUDA streams
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Enhanced with performance optimizations
 */

// ================================================================================================
// SYSTEM INCLUDES AND BASIC DEFINITIONS
// ================================================================================================

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <sys/time.h>

// ================================================================================================
// TIMING AND UTILITY FUNCTIONS
// ================================================================================================

/**
 * High-precision timing function for performance measurement
 * Replaces OpenMP timing with standard system calls
 */
double get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ================================================================================================
// CONSTANTS AND MACROS
// ================================================================================================

#define MAXLINE 2000
#define MAXCAD 200

// Utility macros for min/max operations
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * CUDA error checking macros for robust error handling
 * Essential for debugging GPU kernel launches and memory operations
 */
#define CHECK_CUDA_CALL(a)                                                                            \
	{                                                                                                 \
		cudaError_t ok = a;                                                                           \
		if (ok != cudaSuccess)                                                                        \
			fprintf(stderr, "-- Error CUDA call in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
	}
#define CHECK_CUDA_LAST()                                                                             \
	{                                                                                                 \
		cudaError_t ok = cudaGetLastError();                                                          \
		if (ok != cudaSuccess)                                                                        \
			fprintf(stderr, "-- Error CUDA last in line %d: %s\n", __LINE__, cudaGetErrorString(ok)); \
	}

// ================================================================================================
// CUDA CONSTANT MEMORY DECLARATIONS
// ================================================================================================

/**
 * Constant memory for frequently accessed algorithm parameters
 * Provides broadcast access to all threads with minimal latency
 */
__constant__ int c_numPoints;  // Total number of data points
__constant__ int c_dimensions; // Dimensionality of each point
__constant__ int c_K;		   // Number of clusters

// ================================================================================================
// FILE I/O ERROR HANDLING
// ================================================================================================

/**
 * Display comprehensive file operation error messages
 */
void showFileError(int error, char *filename)
{
	printf("Error\n");
	switch (error)
	{
	case -1:
		fprintf(stderr, "\tFile %s has too many columns.\n", filename);
		fprintf(stderr, "\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
		break;
	case -2:
		fprintf(stderr, "Error reading file: %s.\n", filename);
		break;
	case -3:
		fprintf(stderr, "Error writing file: %s.\n", filename);
		break;
	}
	fflush(stderr);
}

// ================================================================================================
// DATA INPUT/OUTPUT FUNCTIONS
// ================================================================================================

/**
 * Parse input file to determine dataset dimensions
 * Returns: 0 on success, negative error code on failure
 */
int readInput(char *filename, int *lines, int *samples)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int contlines, contsamples = 0;

	contlines = 0;

	if ((fp = fopen(filename, "r")) != NULL)
	{
		while (fgets(line, MAXLINE, fp) != NULL)
		{
			if (strchr(line, '\n') == NULL)
			{
				return -1; // Line too long
			}
			contlines++;
			ptr = strtok(line, delim);
			contsamples = 0;
			while (ptr != NULL)
			{
				contsamples++;
				ptr = strtok(NULL, delim);
			}
		}
		fclose(fp);
		*lines = contlines;
		*samples = contsamples;
		return 0;
	}
	else
	{
		return -2; // File not found
	}
}

/**
 * Load actual data from input file into memory
 * Assumes file dimensions have been validated by readInput()
 */
int readInput2(char *filename, float *data)
{
	FILE *fp;
	char line[MAXLINE] = "";
	char *ptr;
	const char *delim = "\t";
	int i = 0;

	if ((fp = fopen(filename, "rt")) != NULL)
	{
		while (fgets(line, MAXLINE, fp) != NULL)
		{
			ptr = strtok(line, delim);
			while (ptr != NULL)
			{
				data[i] = atof(ptr);
				i++;
				ptr = strtok(NULL, delim);
			}
		}
		fclose(fp);
		return 0;
	}
	else
	{
		return -2; // File not found
	}
}

/**
 * Write final cluster assignments to output file
 */
int writeResult(int *classMap, int lines, const char *filename)
{
	FILE *fp;

	if ((fp = fopen(filename, "wt")) != NULL)
	{
		for (int i = 0; i < lines; i++)
		{
			fprintf(fp, "%d\n", classMap[i]);
		}
		fclose(fp);
		return 0;
	}
	else
	{
		return -3; // Unable to write file
	}
}

// ================================================================================================
// ALGORITHM UTILITY FUNCTIONS
// ================================================================================================

/**
 * Initialize centroids using randomly selected data points
 * Copies actual data points as initial cluster centers
 */
void initCentroids(const float *data, float *centroids, int *centroidPos, int samples, int K)
{
	int i;
	int idx;
	for (i = 0; i < K; i++)
	{
		idx = centroidPos[i];
		memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
	}
}

/**
 * Calculate Euclidean distance between two points
 * Used for CPU-side distance calculations when needed
 */
float euclideanDistance(float *point, float *center, int samples)
{
	float dist = 0.0;
	for (int i = 0; i < samples; i++)
	{
		dist += (point[i] - center[i]) * (point[i] - center[i]);
	}
	dist = sqrt(dist);
	return (dist);
}

/**
 * Matrix initialization utilities
 * Reset arrays to zero for fresh computation
 */
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < columns; j++)
			matrix[i * columns + j] = 0.0;
}

void zeroIntArray(int *array, int size)
{
	int i;
	for (i = 0; i < size; i++)
		array[i] = 0;
}

// ================================================================================================
// CUDA MEMORY MANAGEMENT
// ================================================================================================

/**
 * Configure constant memory with algorithm parameters
 * Provides high-speed broadcast access to all GPU threads
 */
void setConstantMemory(int numPoints, int dimensions, int K)
{
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_numPoints, &numPoints, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_dimensions, &dimensions, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_K, &K, sizeof(int)));
}

// ================================================================================================
// HIGH-PERFORMANCE REDUCTION KERNELS
// ================================================================================================

/**
 * Ultra-fast warp-level reduction for maximum finding
 * Uses register-level shuffle operations for optimal performance
 * Time complexity: O(log₂(32)) = 5 operations per warp
 */
__device__ __forceinline__ float warp_reduce_max(float val)
{
	// Participation mask: all 32 threads in warp participate
	const unsigned int FULL_MASK = 0xffffffff;
#pragma unroll
	for (unsigned int i = 16; i > 0; i /= 2)
	{
		// Exchange values directly through registers (fastest communication)
		val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, i));
	}
	return val;
}

/**
 * Block-level maximum reduction kernel
 * Combines warp-level reductions for maximum scalability
 *
 * ALGORITHM:
 * 1. Grid-stride loop for data loading
 * 2. Intra-warp reduction using shuffle operations
 * 3. Inter-warp reduction via shared memory
 * 4. Single result per block output
 */
__global__ void reduce_max(float *inputs, unsigned int input_size, float *outputs)
{
	// Initialize with neutral element for MAX operation
	float maxVal = -FLT_MAX;

	// ===== PHASE 1: DATA LOADING =====
	// Handle cases where we have fewer threads than data elements
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < input_size;
		 i += blockDim.x * gridDim.x)
		maxVal = fmaxf(maxVal, inputs[i]);

	// ===== PHASE 2: SHARED MEMORY SETUP =====
	// One slot per warp (max 32 warps per block)
	__shared__ float shared[32];

	// Thread positioning within block structure
	unsigned int lane = threadIdx.x % warpSize; // Position within warp (0-31)
	unsigned int wid = threadIdx.x / warpSize;	// Warp ID within block

	// ===== PHASE 3: INTRA-WARP REDUCTION =====
	maxVal = warp_reduce_max(maxVal);
	if (lane == 0)
		shared[wid] = maxVal; // Warp leader writes to shared memory

	__syncthreads(); // Wait for all warps to complete

	// ===== PHASE 4: INTER-WARP REDUCTION =====
	// Create virtual warp from warp leaders
	maxVal = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
	if (wid == 0)
		maxVal = warp_reduce_max(maxVal);

	// ===== PHASE 5: OUTPUT =====
	// Block leader writes final result
	if (threadIdx.x == 0)
		outputs[blockIdx.x] = maxVal;
}

// ================================================================================================
// CORE K-MEANS CUDA KERNELS
// ================================================================================================

/**
 * Calculate Euclidean distances between old and new centroids
 * Used for convergence detection via maximum distance finding
 */
__global__ void calculateCentroidDistances(float *oldCentroids, float *newCentroids,
										   float *distances, int K, int dimensions)
{
	int centroidIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (centroidIdx >= K)
		return;

	float dist = 0.0f;
	// Calculate squared Euclidean distance
	for (int d = 0; d < dimensions; d++)
	{
		int temp_index = centroidIdx * dimensions + d;
		dist = fmaf(oldCentroids[temp_index] - newCentroids[temp_index],
					oldCentroids[temp_index] - newCentroids[temp_index], dist);
	}
	distances[centroidIdx] = sqrtf(dist); // Take square root for actual distance
}

/**
 * High-performance point assignment kernel with shared memory optimization
 *
 * FEATURES:
 * - Dynamic shared memory for centroid caching
 * - Collaborative loading across thread block
 * - Optimized distance calculation using fused multiply-add
 * - Atomic change counting for convergence detection
 */
__global__ void assignPointsToCentroids(
	float *points,
	float *centroids,
	int *assignments,
	int *changes,
	bool useSharedMemory)
{
	extern __shared__ float sharedCentroids[];

	int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pointIdx >= c_numPoints)
		return;

	// ===== SHARED MEMORY OPTIMIZATION =====
	// Collaboratively load centroids into shared memory if enabled
	if (useSharedMemory)
	{
		for (int i = threadIdx.x; i < c_K * c_dimensions; i += blockDim.x)
		{
			if (i < c_K * c_dimensions) // Bounds check
				sharedCentroids[i] = centroids[i];
		}
		__syncthreads();
	}

	// ===== DISTANCE CALCULATION AND ASSIGNMENT =====
	float minDistance = FLT_MAX;
	int bestCentroid = 0;
	int oldAssignment = assignments[pointIdx];

	// Find closest centroid using optimized distance calculation
	for (int k = 0; k < c_K; k++)
	{
		float distance = 0.0f;

		// Calculate squared Euclidean distance (sqrt not needed for comparison)
		for (int d = 0; d < c_dimensions; d++)
		{
			float centroid_val;
			if (useSharedMemory) // Use cached centroids if available
				centroid_val = sharedCentroids[k * c_dimensions + d];
			else
				centroid_val = centroids[k * c_dimensions + d];

			float diff = points[pointIdx * c_dimensions + d] - centroid_val;
			distance = fmaf(diff, diff, distance); // Fused multiply-add for efficiency
		}

		if (distance < minDistance)
		{
			minDistance = distance;
			bestCentroid = k;
		}
	}

	assignments[pointIdx] = bestCentroid + 1; // Convert to 1-based indexing

	// ===== CHANGE DETECTION =====
	// Count assignment changes for convergence monitoring
	if (oldAssignment != bestCentroid + 1)
	{
		atomicAdd(changes, 1);
	}
}

/**
 * Parallel centroid recalculation kernel
 *
 * ORGANIZATION:
 * - Each block handles one cluster (blockIdx.x = cluster ID)
 * - Each thread handles one dimension (threadIdx.x = dimension ID)
 * - Collaborative counting and mean calculation
 *
 * ALGORITHM:
 * 1. Each thread sums points in its assigned cluster/dimension
 * 2. Thread 0 counts total points per cluster
 * 3. Calculate mean coordinates for new centroids
 */
__global__ void recalculateCentroids(
	float *points,
	int *assignments,
	float *newCentroids,
	int *pointsPerCluster)
{
	int clusterIdx = blockIdx.x;
	int dimIdx = threadIdx.x;

	if (clusterIdx >= c_K || dimIdx >= c_dimensions)
		return;

	float sum = 0.0f;
	int count = 0;

	// ===== DATA AGGREGATION =====
	// Each thread handles one dimension of one cluster
	for (int i = 0; i < c_numPoints; i++)
	{
		if (assignments[i] == clusterIdx + 1) // Convert from 1-based indexing
		{
			sum += points[i * c_dimensions + dimIdx];
			if (dimIdx == 0) // Only count once per point to avoid races
				count++;
		}
	}

	// ===== POINT COUNTING =====
	// Store the count (only for dimension 0 to avoid race conditions)
	if (dimIdx == 0)
		pointsPerCluster[clusterIdx] = count;

	__syncthreads(); // Ensure count is available to all threads

	// ===== CENTROID UPDATE =====
	// Calculate mean coordinate for this dimension
	int totalCount = pointsPerCluster[clusterIdx];
	if (totalCount > 0)
		newCentroids[clusterIdx * c_dimensions + dimIdx] = sum / totalCount;
	else
		newCentroids[clusterIdx * c_dimensions + dimIdx] = 0.0f; // Handle empty clusters
}

// ================================================================================================
// MAIN ALGORITHM IMPLEMENTATION
// ================================================================================================

int main(int argc, char *argv[])
{
	// ===== TIMING INITIALIZATION =====
	double start, end;
	start = get_time();

	// ===== PARAMETER VALIDATION =====
	/*
	 * Command line arguments:
	 * argv[1]: Input data file path
	 * argv[2]: Number of clusters (K)
	 * argv[3]: Maximum iterations
	 * argv[4]: Minimum change percentage (termination condition)
	 * argv[5]: Centroid distance threshold (termination condition)
	 * argv[6]: Output file path
	 * argv[7]: Random seed
	 */
	if (argc != 8)
	{
		fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Seed]\n");
		fflush(stderr);
		exit(-1);
	}

	// ===== DATA INPUT AND VALIDATION =====
	int lines = 0, samples = 0;

	int error = readInput(argv[1], &lines, &samples);
	if (error != 0)
	{
		showFileError(error, argv[1]);
		exit(error);
	}

	float *data = (float *)calloc(lines * samples, sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}
	error = readInput2(argv[1], data);
	if (error != 0)
	{
		showFileError(error, argv[1]);
		exit(error);
	}

	// ===== ALGORITHM PARAMETERS =====
	int K = atoi(argv[2]);
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);
	int seed = atoi(argv[7]);

	// ===== MEMORY ALLOCATION =====
	int *centroidPos = (int *)calloc(K, sizeof(int));
	float *centroids = (float *)calloc(K * samples, sizeof(float));
	int *classMap = (int *)calloc(lines, sizeof(int));

	if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	// ===== INITIAL CENTROID SELECTION =====
	srand(seed);
	int i;
	for (i = 0; i < K; i++)
		centroidPos[i] = rand() % lines;

	// Load initial centroids from randomly selected data points
	initCentroids(data, centroids, centroidPos, samples, K);

	// ===== ALGORITHM CONFIGURATION DISPLAY =====
	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);

	end = get_time();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	// ===== CUDA DEVICE INITIALIZATION =====
	CHECK_CUDA_CALL(cudaSetDevice(0));
	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	// Algorithm state variables
	char *outputMsg = (char *)calloc(10000, sizeof(char));
	char line[100];
	int it = 0;
	int changes = 0;
	float maxDist;

	// Working arrays for algorithm execution
	int *pointsPerClass = (int *)malloc(K * sizeof(int));
	float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
	float *distCentroids = (float *)malloc(K * sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	// ===== HIGH-PERFORMANCE GPU COMPUTATION SECTION =====
	/*
	 * This section implements the core K-means algorithm using advanced CUDA optimizations:
	 * - CUDA Events for precise computation timing
	 * - Async memory operations with streams for overlap
	 * - Ultra-fast reduction kernels for convergence detection
	 * - Dynamic shared memory optimization based on device capabilities
	 */

	// ===== CUDA PERFORMANCE MEASUREMENT SETUP =====
	cudaEvent_t computation_start, computation_stop;
	CHECK_CUDA_CALL(cudaEventCreate(&computation_start));
	CHECK_CUDA_CALL(cudaEventCreateWithFlags(&computation_stop, cudaEventBlockingSync));

	cudaStream_t stream;
	CHECK_CUDA_CALL(cudaStreamCreate(&stream));

	// Start high-precision computation timing
	CHECK_CUDA_CALL(cudaEventRecord(computation_start, stream));

	// ===== CUDA CONSTANT MEMORY CONFIGURATION =====
	setConstantMemory(lines, samples, K);

	// ===== GPU MEMORY ALLOCATION =====
	float *d_data, *d_centroids, *d_newCentroids;
	int *d_classMap, *d_changes, *d_pointsPerCluster;

	// Specialized memory for GPU-based maximum distance reduction
	float *d_distCentroids, *d_maxDistResult;
	int paddedK = 1;
	while (paddedK < K)
		paddedK *= 2; // Pad to next power of 2 (required by reduction algorithm)

	// Allocate all GPU memory asynchronously for better performance
	CHECK_CUDA_CALL(cudaMallocAsync(&d_data, lines * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_centroids, K * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_newCentroids, K * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_classMap, lines * sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_changes, sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_pointsPerCluster, K * sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_distCentroids, paddedK * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_maxDistResult, sizeof(float), stream));

	// ===== INITIAL DATA TRANSFER =====
	CHECK_CUDA_CALL(cudaMemcpyAsync(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK_CUDA_CALL(cudaMemcpyAsync(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice, stream));

	// ===== KERNEL LAUNCH CONFIGURATION =====
	dim3 blockSize(32); // Optimized for maximum occupancy
	dim3 gridSize((lines + blockSize.x - 1) / blockSize.x);
	size_t sharedMemSize = K * samples * sizeof(float);
	bool useSharedMemory = true;

	// ===== DEVICE CAPABILITY CHECKING =====
	cudaDeviceProp prop;
	CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
	if (sharedMemSize > prop.sharedMemPerBlock)
	{
		printf("Warning: Shared memory size (%zu bytes) exceeds device limit (%zu bytes)\n",
			   sharedMemSize, prop.sharedMemPerBlock);
		printf("Falling back to global memory access in kernel\n");
		sharedMemSize = 0;
		useSharedMemory = false;
	}

	// ===== MAIN CLUSTERING ITERATION LOOP =====
	do
	{
		it++;
		int zero = 0;

		// ===== KERNEL 1: POINT ASSIGNMENT =====
		// Transfer current centroids and reset change counter
		CHECK_CUDA_CALL(cudaMemcpyAsync(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice, stream));
		CHECK_CUDA_CALL(cudaMemcpyAsync(d_changes, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));

		// Launch point assignment kernel with shared memory optimization
		assignPointsToCentroids<<<gridSize, blockSize, sharedMemSize, stream>>>(
			d_data, d_centroids, d_classMap, d_changes, useSharedMemory);

		// Retrieve assignment results
		CHECK_CUDA_CALL(cudaMemcpyAsync(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_CALL(cudaMemcpyAsync(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost, stream));

		// ===== KERNEL 2: CENTROID RECALCULATION =====
		dim3 centroidGrid(K);		 // One block per cluster
		dim3 centroidBlock(samples); // One thread per dimension

		// Ensure block size doesn't exceed device limits
		if (samples > prop.maxThreadsPerBlock)
		{
			printf("Warning: samples (%d) exceeds max threads per block (%d)\n",
				   samples, prop.maxThreadsPerBlock);
			centroidBlock.x = prop.maxThreadsPerBlock;
		}

		// Launch centroid recalculation kernel
		recalculateCentroids<<<centroidGrid, centroidBlock, 0, stream>>>(
			d_data, d_classMap, d_newCentroids, d_pointsPerCluster);

		// Retrieve updated centroids and cluster information
		CHECK_CUDA_CALL(cudaMemcpyAsync(auxCentroids, d_newCentroids, K * samples * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_CALL(cudaMemcpyAsync(pointsPerClass, d_pointsPerCluster, K * sizeof(int), cudaMemcpyDeviceToHost, stream));

		CHECK_CUDA_CALL(cudaStreamSynchronize(stream)); // Ensure completion before convergence check

		// ===== ULTRA-FAST GPU-BASED CONVERGENCE DETECTION =====
		/*
		 * Replaces CPU bottleneck with high-performance GPU reduction
		 * Uses warp-level primitives for maximum efficiency
		 */

		// Step 1: Calculate distances between old and new centroids
		int distBlockSize = 256;
		int distGridSize = (K + distBlockSize - 1) / distBlockSize;
		calculateCentroidDistances<<<distGridSize, distBlockSize, 0, stream>>>(
			d_centroids, d_newCentroids, d_distCentroids, K, samples);

		// Step 2: Pad distance array for power-of-2 reduction requirement
		if (paddedK > K)
		{
			float negInf = -FLT_MAX;
			CHECK_CUDA_CALL(cudaMemcpyAsync(&d_distCentroids[K], &negInf, (paddedK - K) * sizeof(float),
											cudaMemcpyHostToDevice, stream));
		}

		// Step 3: Apply high-performance reduction to find maximum distance
		int reductionBlockSize = 256;
		int reductionGridSize = (paddedK + reductionBlockSize - 1) / reductionBlockSize;

		if (reductionGridSize == 1)
		{
			// Single block case - direct reduction
			reduce_max<<<1, reductionBlockSize, 0, stream>>>(d_distCentroids, paddedK, d_maxDistResult);
		}
		else
		{
			// Multi-block case - two-stage reduction for optimal performance
			float *d_tempResult;
			CHECK_CUDA_CALL(cudaMallocAsync(&d_tempResult, reductionGridSize * sizeof(float), stream));

			// Stage 1: Multiple blocks to intermediate results
			reduce_max<<<reductionGridSize, reductionBlockSize, 0, stream>>>(
				d_distCentroids, paddedK, d_tempResult);

			// Stage 2: Single warp for final result (optimal for small arrays)
			reduce_max<<<1, 32, 0, stream>>>(d_tempResult, reductionGridSize, d_maxDistResult);

			CHECK_CUDA_CALL(cudaFreeAsync(d_tempResult, stream));
		}

		// Step 4: Retrieve convergence metric
		CHECK_CUDA_CALL(cudaMemcpyAsync(&maxDist, d_maxDistResult, sizeof(float),
										cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_CALL(cudaStreamSynchronize(stream));

		// ===== ITERATION COMPLETION =====
		// Update host centroids for next iteration
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

		sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg, line);

		CHECK_CUDA_CALL(cudaStreamSynchronize(stream));
	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

	// ===== ALGORITHM COMPLETION AND CLEANUP =====

	// Stop computation timing
	CHECK_CUDA_CALL(cudaEventRecord(computation_stop, stream));
	CHECK_CUDA_CALL(cudaEventSynchronize(computation_stop));

	// ===== GPU MEMORY CLEANUP =====
	CHECK_CUDA_CALL(cudaFreeAsync(d_data, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_centroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_newCentroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_classMap, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_changes, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_pointsPerCluster, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_distCentroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_maxDistResult, stream));

	CHECK_CUDA_CALL(cudaStreamDestroy(stream));

	// ===== PERFORMANCE REPORTING =====
	printf("%s", outputMsg);
	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	// Calculate high-precision computation time (CUDA events return milliseconds)
	float computation_time_ms;
	CHECK_CUDA_CALL(cudaEventElapsedTime(&computation_time_ms, computation_start, computation_stop));
	double computation_time = computation_time_ms / 1000.0;
	printf("\nComputation: %f seconds", computation_time);
	fflush(stdout);

	// Cleanup timing resources
	CHECK_CUDA_CALL(cudaEventDestroy(computation_start));
	CHECK_CUDA_CALL(cudaEventDestroy(computation_stop));

	// ===== TIMING OUTPUT FOR ANALYSIS =====
	// Write timing data to file for performance analysis scripts
	char timing_filename[256];
	snprintf(timing_filename, sizeof(timing_filename), "%s.timing", argv[6]);
	FILE *timing_fp = fopen(timing_filename, "w");
	if (timing_fp != NULL)
	{
		fprintf(timing_fp, "computation_time: %f\n", computation_time);
		fclose(timing_fp);
	}

	// ===== TERMINATION CONDITION REPORTING =====
	if (changes <= minChanges)
	{
		printf("\n\nTermination condition:\nMinimum number of changes reached: %d [%d]", changes, minChanges);
	}
	else if (it >= maxIterations)
	{
		printf("\n\nTermination condition:\nMaximum number of iterations reached: %d [%d]", it, maxIterations);
	}
	else
	{
		printf("\n\nTermination condition:\nCentroid update precision reached: %g [%g]", maxDist, maxThreshold);
	}

	// ===== RESULT OUTPUT =====
	error = writeResult(classMap, lines, argv[6]);
	if (error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	// ===== FINAL CLEANUP =====
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	printf("\n\nCUDA K-means completed successfully!\n");
	fflush(stdout);
	return 0;
}
