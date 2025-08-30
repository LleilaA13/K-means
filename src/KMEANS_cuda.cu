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

// CUDA Kernel: Point Assignment
__global__ void assignPointsToCentroids(
	float *points,
	float *centroids,
	int *assignments,
	int *changes)
{
	extern __shared__ float sharedCentroids[];

	int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pointIdx >= c_numPoints)
		return;

	// Load centroids into shared memory (collaborative loading) if shared memory is available
	if (blockDim.x * gridDim.x > 0) // Check if shared memory is allocated
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
			if (blockDim.x * gridDim.x > 0) // Use shared memory if available
				centroid_val = sharedCentroids[k * c_dimensions + d];
			else
				centroid_val = centroids[k * c_dimensions + d];

			float diff = points[pointIdx * c_dimensions + d] - centroid_val;
			distance += diff * diff;
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


__device__ __forceinline__ float warp_reduce(float val)
{
	FULL_MASK = 0xffffffff;
# pragma unroll
	for (unsigned int i = 16; i > 0; i /= 2)
	{
		val = max(val, __shfl_down_sync(FULL_MASK, val, i));
	}
	return val;
}

__global__ void reduce(float* inputs, unsigned int input_size, float* outputs)
{
	/* Eccoci qui all'interno della reduce più veloce del west. Questa implementazione è presa da questo blog:
	 * https://ashvardanian.com/posts/cuda-parallel-reductions/
	 * Praticamente questa implementazione sfrutta delle operazioni che vengono eseguite a livello dei warp. Se vi ricordate, in cuda
	 * i warp sono il più basso livello logico in cui le istruzioni vengono eseguite.
	 * ATTENZIONE: per fare si che questo algoritmo funzioni, input_size DEVE essere una potenza di 2, quindi dovete paddare il vostro array finché non ha
	 * la grandezza desiderata. Questo non influisce sulla correttezza del vostro algoritmo, vi dovete solo ricordare di paddare con un valore neutro per
	 * la vostra operazione (nel caso del MAX il valore è -FLT_MAX oppure semplicemente FLT_MIN)
	 */
    float sum = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < input_size;
            i += blockDim.x * gridDim.x)
        sum += inputs[i]; // Questo for serve in caso non abbiate abbastanza thread per parallelizzare, e quindi ogni thread deve gestire più elementi. Per fortuna non è il vostro caso, quindi questo for in realtà di riduce semplicemente a sum += inputs[i] (fate la prova togliendolo per vedere che effettivamente l'algoritmo funziona lo stesso)

    __shared__ float shared[32];
    unsigned int lane = threadIdx.x % warpSize;
    unsigned int wid = threadIdx.x / warpSize;

    sum = warp_reduce(sum);
    if (lane == 0)
        shared[wid] = sum;

    // Wait for all partial reductions
    __syncthreads();

    sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0)
        sum = warp_reduce(sum);

    if (threadIdx.x == 0)
        outputs[blockIdx.x] = sum;
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
	//**************************************************
	// START CLOCK***************************************
	start = get_time();
	//**************************************************
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

	// Create CUDA events for precise kernel timing
	cudaEvent_t cuda_start, cuda_stop;
	CHECK_CUDA_CALL(cudaEventCreate(&cuda_start));
	CHECK_CUDA_CALL(cudaEventCreateWithFlags(&cuda_stop, cudaEventBlockingSync));

	cudaStream_t stream; // Creati il tuo stream di esecuzione così ti fai la pipeline personalizzata sulla scheda e i trasferimenti in memoria sono fatti meglio perché la scheda usa le mempool. Le operazioni sugli stream sono SERIALI per definizione quindi non ti serve usare le synchronize perché vengono eseguite nell'ordine in cui sono chiamate. Servirà solo una cudaStreamSynchronize(stream) a fine loop te l'ho già messa
	CHECK_CUDA_CALL(cudaStreamCreate(&stream));

	// Set constant memory for CUDA kernels
	setConstantMemory(lines, samples, K);

	// Allocate GPU memory
	float *d_data, *d_centroids, *d_newCentroids;
	int *d_classMap, *d_changes, *d_pointsPerCluster;

	CHECK_CUDA_CALL(cudaMallocAsync(&d_data, lines * samples * sizeof(float), stream)); // Queste malloc e memcpy "async" sono uguali alle altre malloc ma vengono eseguite sul tuo stream, quindi il ritorno è asincrono e non sono bloccanti. Non hai problemi perché tutte le operazioni sono nella tuo stream quindi anche se il ritorno è asincrono sulla scheda l'esecuzione è seriale (nell'ordine in cui le hai mandate)
	CHECK_CUDA_CALL(cudaMallocAsync(&d_centroids, K * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_newCentroids, K * samples * sizeof(float), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_classMap, lines * sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_changes, sizeof(int), stream));
	CHECK_CUDA_CALL(cudaMallocAsync(&d_pointsPerCluster, K * sizeof(int), stream));

	// Copy initial data to GPU
	CHECK_CUDA_CALL(cudaMemcpyAsync(d_data, data, lines * samples * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK_CUDA_CALL(cudaMemcpyAsync(d_classMap, classMap, lines * sizeof(int), cudaMemcpyHostToDevice, stream));

	// Configure kernel launch parameters
	dim3 blockSize(256);
	dim3 gridSize((lines + blockSize.x - 1) / blockSize.x);
	size_t sharedMemSize = K * samples * sizeof(float);

	// Check if shared memory size is within limits
	cudaDeviceProp prop;
	CHECK_CUDA_CALL(cudaGetDeviceProperties(&prop, 0));
	if (sharedMemSize > prop.sharedMemPerBlock)
	{
		printf("Warning: Shared memory size (%zu bytes) exceeds device limit (%zu bytes)\n",
			   sharedMemSize, prop.sharedMemPerBlock);
		printf("Falling back to global memory access in kernel\n");
		sharedMemSize = 0; // Disable shared memory usage
	}

	do
	{
		it++;

		// Copy current centroids to GPU
		CHECK_CUDA_CALL(cudaMemcpyAsync(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice, stream));

		// Reset changes counter
		int zero = 0;
		CHECK_CUDA_CALL(cudaMemcpyAsync(d_changes, &zero, sizeof(int), cudaMemcpyHostToDevice, stream));

		// Record start time for kernel timing
		CHECK_CUDA_CALL(cudaEventRecord(cuda_start, stream));

		// 1. CUDA Kernel: Calculate distances and assign points to centroids
		assignPointsToCentroids<<<gridSize, blockSize, sharedMemSize, stream>>>(
			d_data, d_centroids, d_classMap, d_changes); // Metti lo stream come ultimo argomento al kernel e anche quello viene eseguito sullo stream

		// Record end time for kernel timing
		CHECK_CUDA_CALL(cudaEventRecord(cuda_stop, stream));
		CHECK_CUDA_CALL(cudaEventSynchronize(cuda_stop));

		// Get kernel execution time
		float kernelTime;
		CHECK_CUDA_CALL(cudaEventElapsedTime(&kernelTime, cuda_start, cuda_stop));
		if (it <= 5)
		{
			printf("Iteration %d: Kernel execution time: %.3f ms\n", it, kernelTime);
		}

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

		/* -------------------------- Ciao ragazzi, commentino qui sotto ----------------------- */
		/* Qui c'è il sempiterno problema che fare questa operazione di max con cuda è problematico perché non esiste atomicMax() per i float.
		 * FORTUNATAMENTE puoi usare una strided reduction! L'algoritmo di riduzione funziona perfettamente anche con operazioni tipo max, min, ecc...
		 * In questo link: https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf c'è un pdf dove troverete un'implementazione della
		 * riduzione con svariati livelli di ottimizzazione, ma il vostro bro rick vi ha incluso nella repo un file (reduce.cu) che
		 * contiene un'implementazione ancora più veloce che farà andare in brodo di giuggiole il boss de sensi! All'interno di questa implementazione
		 * troverete una descrizione delle funzioni che vengono usate, così se il de sensi vi chiede qualcosa non cascate dal pero. Considerate di usarla
		 * perché de sensi ha spiegato questa riduzione a lezione (quella che usa il butterfly pattern) e la versione che vi ho dato usa delle cose molto
		 * a basso livello di cuda (operazione collettive all'interno dei thread di un warp). è un pò complicata ma penso che apprezzerà.
		 */
		maxDist = FLT_MIN;
		for (i = 0; i < K; i++)
		{
			distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
			if (distCentroids[i] > maxDist)
			{
				maxDist = distCentroids[i];
			}
		}
		memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

		sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
		outputMsg = strcat(outputMsg, line);

		CHECK_CUDA_CALL(cudaStreamSynchronize(stream));
	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

	// Free GPU memory
	CHECK_CUDA_CALL(cudaFreeAsync(d_data, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_centroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_newCentroids, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_classMap, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_changes, stream));
	CHECK_CUDA_CALL(cudaFreeAsync(d_pointsPerCluster, stream));

	// Cleanup CUDA events
	CHECK_CUDA_CALL(cudaEventDestroy(cuda_start));
	CHECK_CUDA_CALL(cudaEventDestroy(cuda_stop));
	CHECK_CUDA_CALL(cudaStreamDestroy(stream));

	/*
	 *
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */
	// Output and termination conditions
	printf("%s", outputMsg);

	CHECK_CUDA_CALL(cudaDeviceSynchronize());

	// END CLOCK*****************************************
	end = get_time();
	double computation_time = end - start;
	printf("\nComputation: %f seconds", computation_time);
	fflush(stdout);

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


