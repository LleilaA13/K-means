/*
 * k-Means clustering algorithm
 *
 * CUDA version
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <cuda.h>
#include <sys/time.h>

// Simple timing function to replace omp_get_wtime()
double get_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/*
 * Macros to show errors when calling a CUDA library function,
 * or after launching a kernel
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

// Declare constant memory for frequently accessed parameters
__constant__ int c_numPoints;  // Number of data points
__constant__ int c_dimensions; // Number of dimensions per point
__constant__ int c_K;		   // Number of clusters

/*
Function showFileError: It displays the corresponding error during file reading.
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

/*
Function readInput: It reads the file to determine the number of rows and columns.
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
				return -1;
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
		return -2;
	}
}

/*
Function readInput2: It loads data from file.
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
		return -2; // No file found
	}
}

/*
Function writeResult: It writes in the output file the cluster of each sample (point).
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
		return -3; // No file found
	}
}

/*

Function initCentroids: This function copies the values of the initial centroids, using their
position in the input data structure as a reference map.
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

/*
Function euclideanDistance: Euclidean distance
This function could be modified
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

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	int i, j;
	for (i = 0; i < rows; i++)
		for (j = 0; j < columns; j++)
			matrix[i * columns + j] = 0.0;
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	int i;
	for (i = 0; i < size; i++)
		array[i] = 0;
}

// Note: Constant memory already declared at the top of the file

// Host function to set constant memory
void setConstantMemory(int numPoints, int dimensions, int K)
{
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_numPoints, &numPoints, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_dimensions, &dimensions, sizeof(int)));
	CHECK_CUDA_CALL(cudaMemcpyToSymbol(c_K, &K, sizeof(int)));
}

// Efficient warp-level reduction for maximum finding (from Riccardo's suggestion)
__device__ __forceinline__ float warp_reduce_max(float val)
{
	const unsigned int FULL_MASK = 0xffffffff; // Bitmap indicating which threads participate
#pragma unroll
	for (unsigned int i = 16; i > 0; i /= 2)
	{
		// Use shuffle down to exchange values between threads in a warp
		val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, i));
	}
	return val;
}

// GPU reduction kernel for finding maximum value (optimized for K-means centroid distances)
__global__ void reduce_max(float *inputs, unsigned int input_size, float *outputs)
{
	// Initialize with minimum value for maximum operation
	float maxVal = -FLT_MAX;

	// Each thread processes multiple elements if needed (grid-stride loop)
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < input_size;
		 i += blockDim.x * gridDim.x)
		maxVal = fmaxf(maxVal, inputs[i]);

	__shared__ float shared[32]; // Fixed size for up to 1024 threads per block (32 warps max)

	// Warp and thread identification
	unsigned int lane = threadIdx.x % warpSize; // Thread position within warp
	unsigned int wid = threadIdx.x / warpSize;	// Warp ID within block

	// First reduction: within each warp
	maxVal = warp_reduce_max(maxVal);
	if (lane == 0)
		shared[wid] = maxVal; // Warp leader writes to shared memory

	__syncthreads(); // Wait for all warps to complete

	// Second reduction: across warps within block
	maxVal = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
	if (wid == 0)
		maxVal = warp_reduce_max(maxVal);

	// Block leader writes final result
	if (threadIdx.x == 0)
		outputs[blockIdx.x] = maxVal;
}

// GPU kernel to calculate Euclidean distances between old and new centroids
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
		float diff = oldCentroids[centroidIdx * dimensions + d] -
					 newCentroids[centroidIdx * dimensions + d];
		dist += diff * diff;
	}
	distances[centroidIdx] = sqrtf(dist); // Take square root for actual distance
}

// CUDA Kernel: Point Assignment
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

	// Load centroids into shared memory (collaborative loading) if enabled
	if (useSharedMemory)
	{
		for (int i = threadIdx.x; i < c_K * c_dimensions; i += blockDim.x)
		{
			if (i < c_K * c_dimensions) // Bounds check
				sharedCentroids[i] = centroids[i];
		}
		__syncthreads();
	}

	float minDistance = FLT_MAX;
	int bestCentroid = 0;
	int oldAssignment = assignments[pointIdx];

	// Calculate distance to each centroid
	for (int k = 0; k < c_K; k++)
	{
		float distance = 0.0f;

		// Calculate squared Euclidean distance (sqrt not needed for comparison)
		for (int d = 0; d < c_dimensions; d++)
		{
			float centroid_val;
			if (useSharedMemory) // Use shared memory if available
				centroid_val = sharedCentroids[k * c_dimensions + d];
			else
				centroid_val = centroids[k * c_dimensions + d];

			float diff = points[pointIdx * c_dimensions + d] - centroid_val;
			distance = fmaf(diff, diff, distance);
		}

		if (distance < minDistance)
		{
			minDistance = distance;
			bestCentroid = k;
		}
	}

	assignments[pointIdx] = bestCentroid + 1; // Convert to 1-based indexing

	// Count changes using atomic operation
	if (oldAssignment != bestCentroid + 1)
	{
		atomicAdd(changes, 1);
	}
}

// CUDA Kernel: Centroid Recalculation (improved version)
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

	// Each thread handles one dimension of one cluster
	for (int i = 0; i < c_numPoints; i++)
	{
		if (assignments[i] == clusterIdx + 1) // Convert from 1-based
		{
			sum += points[i * c_dimensions + dimIdx];
			if (dimIdx == 0) // Only count once per point
				count++;
		}
	}

	// Store the count (only for dimension 0 to avoid race conditions)
	if (dimIdx == 0)
		pointsPerCluster[clusterIdx] = count;

	__syncthreads();

	// Calculate mean for this dimension
	int totalCount = pointsPerCluster[clusterIdx];
	if (totalCount > 0)
		newCentroids[clusterIdx * c_dimensions + dimIdx] = sum / totalCount;
	else
		newCentroids[clusterIdx * c_dimensions + dimIdx] = 0.0f; // Or keep old centroid
}

int main(int argc, char *argv[])
{

	// START CLOCK***************************************
	double start, end;
	start = get_time();
	//**************************************************
	/*
	 * PARAMETERS
	 *
	 * argv[1]: Input data file
	 * argv[2]: Number of clusters
	 * argv[3]: Maximum number of iterations of the method. Algorithm termination condition.
	 * argv[4]: Minimum percentage of class changes. Algorithm termination condition.
	 *          If between one iteration and the next, the percentage of class changes is less than
	 *          this percentage, the algorithm stops.
	 * argv[5]: Precision in the centroid distance after the update.
	 *          It is an algorithm termination condition. If between one iteration of the algorithm
	 *          and the next, the maximum distance between centroids is less than this precision, the
	 *          algorithm stops.
	 * argv[6]: Output file. Class assigned to each point of the input file.
	 * argv[7]: Seed for random number generation.
	 * */
	if (argc != 8)
	{
		fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Seed]\n");
		fflush(stderr);
		exit(-1);
	}

	// Reading the input data
	// lines = number of points; samples = number of dimensions per point
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

	// Parameters
	int K = atoi(argv[2]);
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(lines * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);
	int seed = atoi(argv[7]);

	int *centroidPos = (int *)calloc(K, sizeof(int));
	float *centroids = (float *)calloc(K * samples, sizeof(float));
	int *classMap = (int *)calloc(lines, sizeof(int));

	if (centroidPos == NULL || centroids == NULL || classMap == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	// Initial centrodis
	srand(seed);
	int i;
	for (i = 0; i < K; i++)
		centroidPos[i] = rand() % lines;

	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);

	printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
	printf("\tNumber of clusters: %d\n", K);
	printf("\tMaximum number of iterations: %d\n", maxIterations);
	printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
	printf("\tMaximum centroid precision: %f\n", maxThreshold);

	// END CLOCK*****************************************
	end = get_time();
	printf("\nMemory allocation: %f seconds\n", end - start);
	fflush(stdout);

	CHECK_CUDA_CALL(cudaSetDevice(0));
	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	char *outputMsg = (char *)calloc(10000, sizeof(char));
	char line[100];

	int it = 0;
	int changes = 0;
	float maxDist;

	// pointPerClass: number of points classified in each class
	// auxCentroids: mean of the points in each class
	int *pointsPerClass = (int *)malloc(K * sizeof(int));
	float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
	float *distCentroids = (float *)malloc(K * sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		exit(-4);
	}

	/*
	 *
	 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	 *
	 */

	/* -------------------- Ciao daniel sono riccardo ecco un pò di chicche tvb ----------------------------- */
	/*
	cudaEvent_t start, stop; // Usa i cudaEvents per prendere i tempi all'interno del programma perché sono il modo migliore e più preciso per beccarti il runtime dei tuoi kernel. Ritornano un valore in millisecondi e ti posso assicurare che sono precisissimi. Di seguito un esempio su come si usano

	CHECK_CUDA(cudaEventCreate(&start))
	CHECK_CUDA(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync))

	CHECK_CUDA(cudaEventRecord(start, stream))
	... kernelozzo <<< ... >>> (argomenti del kernelozzo)
	CHECK_CUDA(cudaEventRecord(stop, stream))
	CHECK_CUDA(cudaEventSynchronize(stop))

	float elapsedTime;
	CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop))

	*/

	// Create CUDA events for precise computation timing
	cudaEvent_t computation_start, computation_stop;
	CHECK_CUDA_CALL(cudaEventCreate(&computation_start));
	CHECK_CUDA_CALL(cudaEventCreateWithFlags(&computation_stop, cudaEventBlockingSync));

	cudaStream_t stream;
	CHECK_CUDA_CALL(cudaStreamCreate(&stream));

	// START COMPUTATION TIMING WITH CUDA EVENTS
	CHECK_CUDA_CALL(cudaEventRecord(computation_start, stream));

	// Set constant memory for CUDA kernels
	setConstantMemory(lines, samples, K);

	// Allocate GPU memory
	float *d_data, *d_centroids, *d_newCentroids;
	int *d_classMap, *d_changes, *d_pointsPerCluster;

	// Additional memory for GPU-based maxDist calculation (Riccardo's optimization)
	float *d_distCentroids, *d_maxDistResult;
	int paddedK = 1;
	while (paddedK < K)
		paddedK *= 2; // Pad to next power of 2 as required by reduction

	CHECK_CUDA_CALL(cudaMallocAsync(&d_data, lines * samples * sizeof(float), stream)); // Queste malloc e memcpy "async" sono uguali alle altre malloc ma vengono eseguite sul tuo stream, quindi il ritorno è asincrono e non sono bloccanti. Non hai problemi perché tutte le operazioni sono nella tuo stream quindi anche se il ritorno è asincrono sulla scheda l'esecuzione è seriale (nell'ordine in cui le hai mandate)
	CHECK_CUDA_CALL(cudaMallocAsync(&d_centroids, K * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_newCentroids, K * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_classMap, lines * sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_changes, sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_pointsPerCluster, K * sizeof(int), stream));

	// Allocate memory for distance calculation and reduction
	CHECK_CUDA_CALL(cudaMallocAsync(&d_distCentroids, paddedK * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_maxDistResult, sizeof(float), stream));

	// Copy initial data to GPU
	CHECK_CUDA_CALL(cudaMemcpyAsync(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK_CUDA_CALL(cudaMemcpyAsync(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice, stream));

	dim3 blockSize(32);
	dim3 gridSize((lines + blockSize.x - 1) / blockSize.x);
	size_t sharedMemSize = K * samples * sizeof(float);
	bool useSharedMemory = true;

	// Check if shared memory size is within limits
	cudaDeviceProp prop;
	CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
	if (sharedMemSize > prop.sharedMemPerBlock)
	{
		printf("Warning: Shared memory size (%zu bytes) exceeds device limit (%zu bytes)\n",
			   sharedMemSize, prop.sharedMemPerBlock);
		printf("Falling back to global memory access in kernel\n");
		sharedMemSize = 0; // Disable shared memory usage
		useSharedMemory = false;
	}

	do
	{
		it++;
		// Reset changes counter
		int zero = 0;

		// Copy current centroids to GPU
		CHECK_CUDA_CALL(cudaMemcpyAsync(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice, stream));
		CHECK_CUDA_CALL(cudaMemcpyAsync(d_changes, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));

		// 1. CUDA Kernel: Calculate distances and assign points to centroids
		assignPointsToCentroids<<<gridSize, blockSize, sharedMemSize, stream>>>(
			d_data, d_centroids, d_classMap, d_changes, useSharedMemory);

		// Copy results back to host
		CHECK_CUDA_CALL(cudaMemcpyAsync(classMap, d_classMap, lines * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_CALL(cudaMemcpyAsync(&changes, d_changes, sizeof(int), cudaMemcpyDeviceToHost, stream));

		// 2. CUDA Kernel: Recalculate centroids on GPU
		dim3 centroidGrid(K);
		dim3 centroidBlock(samples);

		// Make sure block size doesn't exceed device limits
		if (samples > prop.maxThreadsPerBlock)
		{
			printf("Warning: samples (%d) exceeds max threads per block (%d)\n",
				   samples, prop.maxThreadsPerBlock);
			centroidBlock.x = prop.maxThreadsPerBlock;
		}

		recalculateCentroids<<<centroidGrid, centroidBlock, 0, stream>>>(
			d_data, d_classMap, d_newCentroids, d_pointsPerCluster);

		// Copy new centroids back to host for convergence check
		CHECK_CUDA_CALL(cudaMemcpyAsync(auxCentroids, d_newCentroids, K * samples * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_CALL(cudaMemcpyAsync(pointsPerClass, d_pointsPerCluster, K * sizeof(int), cudaMemcpyDeviceToHost, stream));

		CHECK_CUDA_CALL(cudaStreamSynchronize(stream)); // Wait for GPU operations to complete

		/* -------------------------- GPU-based maxDist calculation using Riccardo's reduction ----------------------- */
		/* Replacing the CPU bottleneck with ultra-fast GPU reduction using warp-level operations */

		// Step 1: Calculate centroid distances on GPU
		int distBlockSize = 256;
		int distGridSize = (K + distBlockSize - 1) / distBlockSize;
		calculateCentroidDistances<<<distGridSize, distBlockSize, 0, stream>>>(
			d_centroids, d_newCentroids, d_distCentroids, K, samples);

		// Step 2: Pad the distance array with -FLT_MAX for reduction (requirement for power of 2)
		if (paddedK > K)
		{
			float negInf = -FLT_MAX;
			for (int i = K; i < paddedK; i++)
			{
				CHECK_CUDA_CALL(cudaMemcpyAsync(&d_distCentroids[i], &negInf, sizeof(float),
												cudaMemcpyHostToDevice, stream));
			}
		}

		// Step 3: Apply reduction to find maximum distance
		// Following Riccardo's instructions for single vs multi-block reduction
		int reductionBlockSize = 256;
		int reductionGridSize = (paddedK + reductionBlockSize - 1) / reductionBlockSize;

		if (reductionGridSize == 1)
		{
			// Single block case - direct reduction
			reduce_max<<<1, reductionBlockSize, 0, stream>>>(d_distCentroids, paddedK, d_maxDistResult);
		}
		else
		{
			// Multi-block case - two-stage reduction as per Riccardo's instructions
			float *d_tempResult;
			CHECK_CUDA_CALL(cudaMallocAsync(&d_tempResult, reductionGridSize * sizeof(float), stream));

			// First reduction: multiple blocks to intermediate results
			reduce_max<<<reductionGridSize, reductionBlockSize, 0, stream>>>(
				d_distCentroids, paddedK, d_tempResult);

			// Second reduction: single block for final result (using warp size as recommended)
			reduce_max<<<1, 32, 0, stream>>>(d_tempResult, reductionGridSize, d_maxDistResult);

			CHECK_CUDA_CALL(cudaFreeAsync(d_tempResult, stream));
		}

		// Step 4: Copy result back to host
		CHECK_CUDA_CALL(cudaMemcpyAsync(&maxDist, d_maxDistResult, sizeof(float),
										cudaMemcpyDeviceToHost, stream));
		CHECK_CUDA_CALL(cudaStreamSynchronize(stream)); // Ensure completion before using maxDist

		// Update centroids on host (still needed for next iteration)
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

		sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg, line);

		CHECK_CUDA_CALL(cudaStreamSynchronize(stream));
	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */

	// END COMPUTATION TIMING WITH CUDA EVENTS
	CHECK_CUDA_CALL(cudaEventRecord(computation_stop, stream));
	CHECK_CUDA_CALL(cudaEventSynchronize(computation_stop));

	// Free GPU memory
	CHECK_CUDA_CALL(cudaFreeAsync(d_data, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_centroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_newCentroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_classMap, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_changes, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_pointsPerCluster, stream));

	// Free additional GPU memory for reduction
	CHECK_CUDA_CALL(cudaFreeAsync(d_distCentroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_maxDistResult, stream));

	CHECK_CUDA_CALL(cudaStreamDestroy(stream));

	// Output and termination conditions
	printf("%s", outputMsg);

	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	// Calculate computation time using CUDA events (returns milliseconds, convert to seconds)
	float computation_time_ms;
	CHECK_CUDA_CALL(cudaEventElapsedTime(&computation_time_ms, computation_start, computation_stop));
	double computation_time = computation_time_ms / 1000.0;
	printf("\nComputation: %f seconds", computation_time);
	fflush(stdout);

	// Cleanup computation timing events
	CHECK_CUDA_CALL(cudaEventDestroy(computation_start));
	CHECK_CUDA_CALL(cudaEventDestroy(computation_stop));

	// Write timing to file for consistency with other implementations
	char timing_filename[256];
	snprintf(timing_filename, sizeof(timing_filename), "%s.timing", argv[6]);
	FILE *timing_fp = fopen(timing_filename, "w");
	if (timing_fp != NULL)
	{
		fprintf(timing_fp, "computation_time: %f\n", computation_time);
		fclose(timing_fp);
	}

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

	// Writing the classification of each point to the output file.
	error = writeResult(classMap, lines, argv[6]);
	if (error != 0)
	{
		showFileError(error, argv[6]);
		exit(error);
	}

	// Free memory
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
