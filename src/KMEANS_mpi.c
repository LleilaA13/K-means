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
		dist = fmaf(point[i] - center[i], point[i] - center[i], dist);
	}
	return dist; // Squared distance
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
	// Initialize MPI:
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	// START CLOCK***************************************
	double start_time = MPI_Wtime();
	double end_time;
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
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Reading the input data
	// lines = number of points (rows); samples = number of dimensions per point (columns)
	int lines = 0, samples = 0;

	int error = readInput(argv[1], &lines, &samples);
	if (error != 0)
	{
		showFileError(error, argv[1]);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	float *data = (float *)calloc(lines * samples, sizeof(float));
	if (data == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}
	error = readInput2(argv[1], data);
	if (error != 0)
	{
		showFileError(error, argv[1]);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
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
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	// Initial centroids
	srand(seed);
	int i;
	for (i = 0; i < K; i++)
		centroidPos[i] = rand() % lines;

	// Loading the array of initial centroids with the data from the array data
	// The centroids are points stored in the data array.
	initCentroids(data, centroids, centroidPos, samples, K);

	if (rank == 0)
	{
		printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
		printf("\tNumber of clusters: %d\n", K);
		printf("\tMaximum number of iterations: %d\n", maxIterations);
		printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
		printf("\tMaximum centroid precision: %f\n", maxThreshold);
	}

	// END CLOCK*****************************************
	end_time = MPI_Wtime();
	printf("\nMemory allocation: %f seconds\n", end_time - start_time);
	fflush(stdout);
	//**************************************************
	// START CLOCK***************************************
	start_time = MPI_Wtime();
	//**************************************************
	char *outputMsg = (char *)calloc(10000, sizeof(char));
	char line[100];

	int j;
	int class;
	float dist, minDist;
	int it = 0;
	int changes = 0;
	float float_changes = 0.0f; 
	float maxDist;

	float *pointsPerClass = (float *)malloc(K * sizeof(float)); 
	float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
	float *distCentroids = (float *)malloc(K * sizeof(float));
	if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
	{
		fprintf(stderr, "Memory allocation error.\n");
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/*
	 *
	 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
	 *
	 */

	// number of points not divisible by number of processes:
	int local_lines = lines / size;
	int remainder = lines % size;

	printf("rank: %d, remainder: %d, local_lines = %d\n", rank, remainder, local_lines);

	// the first remainder ranks each get one extra row if rank < remainder:
	if (rank < remainder)
	{
		local_lines++;
	}

	int start_index;
	
	start_index = (rank * local_lines * samples) / 100;



	printf("rank %d here, start_index = %d\n", rank, start_index);

	size_t allgather_buffer_size = (1 + K + (K * samples)); 
	float *allgather_buffer = (float *)malloc(allgather_buffer_size * sizeof(float));

	do
	{
		it++;

		// 1. Calculate the distance from each point to the centroid
		// Assign each point to the nearest centroid.
		changes = 0;

		for (i = start_index; i < start_index + local_lines; ++i)
		{
			class = 1;
			minDist = FLT_MAX;
			for (j = 0; j < K; ++j)
			{
				dist = euclideanDistance(&data[i * samples], &centroids[j * samples], samples);

				if (dist < minDist)
				{
					minDist = dist;
					class = j + 1;
				}
			}
			if (classMap[i] != class)
			{
				changes++;
			}
			classMap[i] = class;
		}

		// 2. Recalculates the centroids: calculates the mean within each cluster
		// zeroIntArray(pointsPerClass, K);
		zeroFloatMatriz(pointsPerClass, K, 1);
		zeroFloatMatriz(auxCentroids, K, samples);



		for (i = start_index; i < start_index + local_lines; ++i)
		{
			class = classMap[i];
			pointsPerClass[class - 1] += 1.0f;
			for (j = 0; j < samples; j++)
			{
				auxCentroids[(class - 1) * samples + j] += data[i * samples + j];
			}
		}

		float_changes = (float)changes;
		allgather_buffer[0] = float_changes;
		memcpy(&allgather_buffer[1], pointsPerClass, K * sizeof(float));
		memcpy(&allgather_buffer[1 + K], auxCentroids, K * samples * sizeof(float));

		MPI_Allreduce(MPI_IN_PLACE, allgather_buffer, allgather_buffer_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

		changes = (int)allgather_buffer[0];
		memcpy(pointsPerClass, &allgather_buffer[1], K * sizeof(float));
		memcpy(auxCentroids, &allgather_buffer[1 + K], K * samples * sizeof(float));

		for (i = 0; i < K; i++)
		{
			for (j = 0; j < samples; j++)
			{
				auxCentroids[i * samples + j] /= pointsPerClass[i]; 
			}
		}

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

		if (rank == 0)
		{
			sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
			outputMsg = strcat(outputMsg, line);
		}

	} while ((changes > minChanges) && (it < maxIterations) && (maxDist > pow(maxThreshold, 2)));


	int *recvcounts = (int *)malloc(size * sizeof(int)); 
	int *displs = (int *)malloc(size * sizeof(int));	 

	MPI_Gather(&local_lines, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		int total_lines = recvcounts[0]; 
		displs[0] = 0;					 
		for (int i = 1; i < size; i++)
		{
			displs[i] = displs[i - 1] + recvcounts[i - 1]; 
			total_lines += recvcounts[i];				   
		}
		// Check if the total lines match the expected lines:
		if (total_lines != lines)
		{
			fprintf(stderr, "Error: Mismatched line counts (expected %d, got %d)\n", lines, total_lines);
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
	}

	MPI_Gatherv(&classMap[start_index], local_lines, MPI_INT, classMap, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
	free(recvcounts);
	free(displs);
	/*
	 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
	 *
	 */
	// Output and termination conditions
	printf("%s", outputMsg);

	// END CLOCK*****************************************
	end_time = MPI_Wtime();
	double computation_time = end_time - start_time;
	printf("\nComputation: %f seconds", computation_time);
	fflush(stdout);

	// Write timing to file (only rank 0)
	if (rank == 0)
	{
		char timing_filename[256];
		snprintf(timing_filename, sizeof(timing_filename), "%s.timing", argv[6]);
		FILE *timing_fp = fopen(timing_filename, "w");
		if (timing_fp != NULL)
		{
			fprintf(timing_fp, "computation_time: %f\n", computation_time);
			fclose(timing_fp);
		}
	}
	//**************************************************
	// START CLOCK***************************************
	start_time = MPI_Wtime();
	//**************************************************

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

	
	if (rank == 0)
	{
		error = writeResult(classMap, lines, argv[6]);
		if (error != 0)
		{
			showFileError(error, argv[6]);
			
		}
	}

	// Free memory
	free(data);
	free(classMap);
	free(centroidPos);
	free(centroids);
	free(distCentroids);
	free(pointsPerClass);
	free(auxCentroids);

	// free(local_data);
	// free(local_classMap);
	// free(local_pointsPerClass);
	// free(local_auxCentroids);

	// END CLOCK*****************************************
	end_time = MPI_Wtime();
	printf("\n\nMemory deallocation: %f seconds\n", end_time - start_time);
	fflush(stdout);
	//***************************************************/
	MPI_Finalize();
	return 0;
}
