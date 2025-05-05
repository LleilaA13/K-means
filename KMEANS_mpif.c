/*
 * k-Means clustering algorithm
 *
 * MPI version
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
#include <mpi.h>

#define MAXLINE 2000
#define MAXCAD 200

// Macros
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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
	return dist; // Squared Distance
}

/*
Function zeroFloatMatriz: Set matrix elements to 0
This function could be modified
*/
void zeroFloatMatriz(float *matrix, int rows, int columns)
{
	memset(matrix, 0, rows * columns * sizeof(float));
}

/*
Function zeroIntArray: Set array elements to 0
This function could be modified
*/
void zeroIntArray(int *array, int size)
{
	memset(array, 0, size * sizeof(int));
}

int main(int argc, char *argv[])
{
	// Initialize MPI
	MPI_Init(&argc, &argv);
	int rank, size;
	// Get the rank of the current process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// Get the total number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// Set the error handler for MPI_COMM_WORLD to return errors instead of aborting
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	// START CLOCK***************************************
	double start, end;
	start = MPI_Wtime();
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
	 * */
	if (argc != 7)
	{
		fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
		fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Reading the input data on the root process (rank 0)
	// lines = number of points; samples = number of dimensions per point
	int N = 0, D = 0;
	float *points = NULL;

	if (rank == 0)
	{
		int error = readInput(argv[1], &N, &D);
		if (error != 0)
		{
			showFileError(error, argv[1]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		points = (float *)calloc(N * D, sizeof(float));
		if (points == NULL)
		{
			fprintf(stderr, "Memory allocation error.\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		error = readInput2(argv[1], points);
		if (error != 0)
		{
			showFileError(error, argv[1]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	// Broadcast the values of lines (data points) and samples (dimensions) to all processes
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// Everyone gets the arguments of the program
	int K = atoi(argv[2]);
	int maxIterations = atoi(argv[3]);
	int minChanges = (int)(N * atof(argv[4]) / 100.0);
	float maxThreshold = atof(argv[5]);

	float *centroids = (float *)calloc(K * D, sizeof(float));
	if (centroids == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	int *classMap = NULL;

	// Rank 0 must initialize centroids and class mappings, all other processes will get the arrays from it
	if (rank == 0)
	{
		int *centroidPos = (int *)calloc(K, sizeof(int));
		classMap = (int *)calloc(N, sizeof(int));

		if (centroidPos == NULL || classMap == NULL)
		{
			fprintf(stderr, "Memory allocation error.\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		srand(0);
		for (int i = 0; i < K; i++)
			centroidPos[i] = rand() % N;

		// Loading the array of initial centroids with the data from the array data
		// The centroids are points stored in the data array.
		initCentroids(points, centroids, centroidPos, D, K);
		free(centroidPos);

		printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], N, D);
		printf("\tNumber of clusters: %d\n", K);
		printf("\tMaximum number of iterations: %d\n", maxIterations);
		printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), N);
		printf("\tMaximum centroid precision: %f\n", maxThreshold);
	}

	// Broadcast the centroids to all the processes
	MPI_Bcast(centroids, K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

	// END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n%d |Memory allocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	MPI_Barrier(MPI_COMM_WORLD); // Ensure that all processes start timer at the same time
	start = MPI_Wtime();
	//**************************************************

	char *outputMsg = NULL;
	//	char* line;

	if (rank == 0)
	{
		outputMsg = (char *)calloc(10000, sizeof(char));
		//		line = (char*) calloc(100, sizeof(char));
	}

	int it = 0;
	int changes;
	float maxDist;
	int *pointsPerClass = (int *)malloc(K * sizeof(int));
	float *auxCentroids = (float *)malloc(K * D * sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	//  VALUES NEEDED FOR STEP 1: Distribute data points among processes
	int *sendcounts = (int *)malloc(size * sizeof(int));
	int *displs = (int *)malloc(size * sizeof(int));
	int remainder = N % size;
	int sum = 0;
	for (int i = 0; i < size; ++i)
	{
		sendcounts[i] = (N / size) * D;
		if (i < remainder)
			sendcounts[i] += D; // Distribute the remainder among the first 'remainder' processes
		displs[i] = sum;
		sum += sendcounts[i];
	}

	// Works also with odd number of processes / points
	// Calculate the number of local lines (data points) for each process
	int local_n = sendcounts[rank] / D;
	// Allocate memory for local data points and their class assignments
	float *local_points = (float *)calloc(local_n * D, sizeof(float));
	int *local_classMap = (int *)calloc(local_n, sizeof(int));
	if (local_points == NULL || local_classMap == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Scatter the data points from the root process to all processes
	// MPI_Scatterv allows varying counts of data to be sent to each process
	MPI_Scatterv(points, sendcounts, displs, MPI_FLOAT, local_points, sendcounts[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);

	//  VALUES NEEDED FOR STEP 2: Distribute centroid updates among processes
	int *centroid_sendcounts = (int *)malloc(size * sizeof(int));
	int *centroid_displs = (int *)malloc(size * sizeof(int));
	int centroid_remainder = K % size;
	sum = 0;
	for (int i = 0; i < size; ++i)
	{
		centroid_sendcounts[i] = (K / size) * D;
		if (i < centroid_remainder)
			centroid_sendcounts[i] += D; // Distribute remainder centroids
		centroid_displs[i] = sum;
		sum += centroid_sendcounts[i];
	}

	int local_k = centroid_sendcounts[rank] / D; // Number of centroids handled by this process
	// Allocate memory for local centroid updates
	float *local_centroids = (float *)calloc(local_k * D, sizeof(float));
	if (local_centroids == NULL) {
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	do
	{
		it++; // Increment iteration counter

		/* -------------------------------------------------------------------
		 *  STEP 1: Assign points to nearest centroid
		 *	Calculate the distance from each point to the centroid
		 *	Assign each point to the nearest centroid.
		 ------------------------------------------------------------------- */
		
		int local_changes = 0; // counter for changes in cluster assignments, local to each process
		
		// For each local point...
		for (int i = 0; i < local_n; i++)
		{
			int class = 1;
			float minDist = FLT_MAX;

			// For each centroid...
			for (int j = 0; j < K; j++)
			{
				// Compute l_2 (squared, without sqrt)
				float dist = euclideanDistance(&local_points[i * D], &centroids[j * D], D);

				// If the distance is smallest so far, update minDist and the class of the point
				if (dist < minDist)
				{
					minDist = dist;
					class = j + 1;
				}
			}

			// If the class changed, increment the local change counter
			if (local_classMap[i] != class)
			{
				local_changes++;
			}

			// Assign the new class to the point
			local_classMap[i] = class;
		}

		// Gather all the changes from each process and sum them up
		MPI_Request MPI_REQUEST; // Handle for the non-blocking reduction
		// MPI_Iallreduce initiates a non-blocking reduction operation where all processes contribute
		// their local_changes, and the sum is stored in 'changes' for all the process
		MPI_Iallreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &MPI_REQUEST);
		
		/* -------------------------------------------------------------------
		 *    STEP 2: Recalculate centroids (cluster means)
		 ------------------------------------------------------------------- */

		// Initialize pointsPerClass and the centroid auxiliary tables
		zeroIntArray(pointsPerClass, K);		   // Reset cluster counts
		zeroFloatMatriz(auxCentroids, K, D); // Reset centroid accumulator

		// Sum the coordinate of all local points
		for (int i = 0; i < local_n; i++)
		{
			int class = local_classMap[i];
			pointsPerClass[class - 1]++;
			for (int j = 0; j < D; j++)
			{
				auxCentroids[(class - 1) * D + j] += local_points[i * D + j];
			}
		}

		// All the processes receive the other pointsPerClass and auxiliary centroids
		// Reduce all pointsPerClass and auxCentroids across all processes
		// MPI_Allreduce sums up the pointsPerClass and auxCentroids from all processes
		MPI_Allreduce(MPI_IN_PLACE, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, auxCentroids, K * D, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		/* -------------------------------------------------------------------
		*  STEP 3: Check convergence
		*  Compute the maximum distance between old and new centroids
		 ------------------------------------------------------------------- */

		float local_maxDist = 0.0f;

		// For each local centroid handled by this process...
		for (int i = 0; i < local_k; i++)
		{	
			// Calculate the global index of the centroid, used for querying the global centroids table
			// Used for querying the global centroids table
			int global_idx = centroid_displs[rank] / D + i;
			if (global_idx >= K)
				break;

			float distance = 0.0f;

			if (pointsPerClass[global_idx] == 0)
				continue;
			
			// For each dimension...
			for (int j = 0; j < D; j++)
			{
				// Compute the new centroid value by averaging the coordinates
				float centroid_val = auxCentroids[global_idx * D + j] / pointsPerClass[global_idx];
				// Compute the difference with the previous value
				float diff = local_centroids[i * D + j] - centroid_val;
				distance += diff * diff;
				// Update the local centroid with the new value (coordinate)
				local_centroids[i * D + j] = centroid_val;
			}

			// Update the local maximum distance if necessary, for later convergence check
			if (distance > local_maxDist)
			{
				local_maxDist = distance;
			}
		}

		// Reduce to find the maximum distance across all processes
		MPI_Allreduce(&local_maxDist, &maxDist, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

		// Wait for the non-blocking reduction to complete
		
		// Gather all local centroids into the global centroids array
		MPI_Allgatherv(local_centroids, local_k * D, MPI_FLOAT, centroids, centroid_sendcounts, centroid_displs, MPI_FLOAT, MPI_COMM_WORLD);
		// MPI_Allgatherv gathers variable amounts of data from all processes and distributes
		// the combined data to all processes. This updates the centroids for the next iteration.
		
		// Wait if the non-blocking reduction didn't complete
		MPI_Wait(&MPI_REQUEST, MPI_STATUS_IGNORE);

	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold*maxThreshold));

	// Prepare to gather the class assignments from all processes
	int *recvcounts = (int *)malloc(size * sizeof(int));
	int *rdispls = (int *)malloc(size * sizeof(int));
	sum = 0;
	for (int i = 0; i < size; ++i)
	{
		recvcounts[i] = sendcounts[i] / D; // Number of points per process
		rdispls[i] = sum;
		sum += recvcounts[i];
	}

	// Gather all local_classMap arrays from each process into the global classMap array on the root process
	MPI_Gatherv(local_classMap, local_n, MPI_INT, classMap, recvcounts, rdispls, MPI_INT, 0, MPI_COMM_WORLD);
	
	// 	Output and termination conditions
	if (rank == 0)
	{
		printf("%s", outputMsg);
	}

	// END CLOCK*****************************************
	end = MPI_Wtime();
	// Reduce to get the maximum time across all processes
	double computation_time = end - start;
	double max_computation_time;
	MPI_Reduce(&computation_time, &max_computation_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	// Thread 0 print the maximum computation time
	if (rank == 0)
	{
		printf("\n Computation: %f seconds\n", max_computation_time);
		fflush(stdout);
	}
	//**************************************************
	// START CLOCK***************************************
	MPI_Barrier(MPI_COMM_WORLD); // Ensure that all processes start timer at the same time
	start = MPI_Wtime();
	//**************************************************

	if (rank == 0)
	{
		if (changes <= minChanges)
		{
			printf("\n\nTermination condition: Minimum number of changes reached: %d [%d]", changes, minChanges);
		}
		else if (it >= maxIterations)
		{
			printf("\n\nTermination condition: Maximum number of iterations reached: %d [%d]", it, maxIterations);
		}
		else
		{
			printf("\n\nTermination condition: Centroid update precision reached: %g [%g]", maxDist, maxThreshold);
		}

		int error = writeResult(classMap, N, argv[6]);
		if (error != 0)
		{
			showFileError(error, argv[6]);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
		fflush(stdout);
	}

	//	FREE LOCAL ARRAYS: Free memory allocated for each process
	free(local_points);
	free(local_classMap);
	free(local_centroids);
	free(sendcounts);
	free(displs);
	free(centroid_sendcounts);
	free(centroid_displs);
	free(recvcounts);
	free(rdispls);

	//	Free memory on the root process
	if (rank == 0)
	{
		free(points);
		free(classMap);
		free(outputMsg);
	}

	free(centroids);
	free(pointsPerClass);
	free(auxCentroids);

	// END CLOCK*****************************************
	end = MPI_Wtime();
	printf("\n\n%d |Memory deallocation: %f seconds\n", rank, end - start);
	fflush(stdout);
	//***************************************************/

	//	FINALIZE: Clean up the MPI environment
	MPI_Finalize();
	return 0;
}