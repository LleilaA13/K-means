/*
 * k-Means clustering algorithm
 *
 * MPI + OpenMP version
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
#include <omp.h>

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
    float maxDist;

    // pointPerClass: number of points classified in each class
    // auxCentroids: mean of the points in each class
    int *pointsPerClass = (int *)malloc(K * sizeof(int));
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
    // Defining the local variables ecc for each process to work with:
    int local_changes = 0;
    int *local_pointsPerClass = (int *)malloc(K * sizeof(int));
    float *local_auxCentroids = (float *)malloc(K * samples * sizeof(float));

    // number of points not divisible by number of processes:
    int local_lines = lines / size;
    int remainder = lines % size;

    // the first remainder ranks each get one extra row if rank < remainder:
    if (rank < remainder)
    {
        local_lines++;
    }

    int start_index;
    if (rank < remainder)
    {
        start_index = rank * local_lines * samples; // start_index is the starting element index, not row
    }
    else
    {
        start_index = remainder * (local_lines + 1) * samples + (rank - remainder) * local_lines * samples;
    }

    // Defining local data
    int *local_classMap = (int *)calloc(local_lines, sizeof(int));
    float *local_data = (float *)malloc(local_lines * samples * sizeof(float));

    for (i = 0; i < local_lines * samples; i++)
    {
        local_data[i] = data[start_index + i];
    }

    do
    {
        it++;

        // 1. Calculate the distance from each point to the centroid
        // Assign each point to the nearest centroid.
        changes = 0;
        local_changes = 0;

// Parallelize assignment step
#pragma omp parallel for private(j, class, minDist, dist) reduction(+ : local_changes)
        for (i = 0; i < local_lines; i++)
        {
            class = 1;
            minDist = FLT_MAX;
            for (j = 0; j < K; j++)
            {
                dist = euclideanDistance(&local_data[i * samples], &centroids[j * samples], samples);

                if (dist < minDist)
                {
                    minDist = dist;
                    class = j + 1;
                }
            }
            if (local_classMap[i] != class)
            {
                local_changes++;
            }
            local_classMap[i] = class;
        }

        // once we are done with the computation of the distance and updating local_classMap, we
        // can reduce it to changes:
        MPI_Request req;
        MPI_Iallreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &req);
        // MPI_Allreduce(&local_changes, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 2. Recalculates the centroids: calculates the mean within each cluster
        zeroIntArray(pointsPerClass, K);
        zeroFloatMatriz(auxCentroids, K, samples);

        memset(local_pointsPerClass, 0, K * sizeof(int));           // Initialize local points per class
        memset(local_auxCentroids, 0, K * samples * sizeof(float)); // Initialize local auxiliary centroids

// Parallelize local centroid accumulation
#pragma omp parallel for private(j, class)
        for (i = 0; i < local_lines; i++)
        {
            class = local_classMap[i];
#pragma omp atomic
            local_pointsPerClass[class - 1] += 1;
            for (j = 0; j < samples; j++)
            {
#pragma omp atomic
                local_auxCentroids[(class - 1) * samples + j] += local_data[i * samples + j];
            }
        }
        // Update the global pointsPerClass and auxCentroids: BLOCKING !!
        MPI_Allreduce(local_pointsPerClass, pointsPerClass, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_auxCentroids, auxCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

// Parallelize centroid update
#pragma omp parallel for private(j)
        for (i = 0; i < K; i++)
        {
            for (j = 0; j < samples; j++)
            {
                if (pointsPerClass[i] > 0)
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

        MPI_Wait(&req, MPI_STATUS_IGNORE); // Wait for the reduce operation to complete
        if (rank == 0)
        {
            sprintf(line, "\n[%d] Cluster changes: %d\tMax. centroid distance: %f", it, changes, maxDist);
            outputMsg = strcat(outputMsg, line);
        }

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > pow(maxThreshold, 2)));
    // Gather classMaps from all processes. but the processes have different local_classMap sizes
    // use Gatherv
    int *recvcounts = (int *)malloc(size * sizeof(int)); // Number of elements to receive from each process
    int *displs = (int *)malloc(size * sizeof(int));     // Displacements for each process

    MPI_Gather(&local_lines, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD); // Gather local line counts

    if (rank == 0)
    {
        int total_lines = recvcounts[0]; // Initialize total_lines with the first process's count
        displs[0] = 0;                   // Initialize displacements
        for (int i = 1; i < size; i++)
        {
            displs[i] = displs[i - 1] + recvcounts[i - 1]; // Calculate displacements
            total_lines += recvcounts[i];                  // Update total_lines
        }
        // Check if the total lines match the expected lines:
        if (total_lines != lines)
        {
            fprintf(stderr, "Error: Mismatched line counts (expected %d, got %d)\n", lines, total_lines);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Gatherv(local_classMap, local_lines, MPI_INT, classMap, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

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

    // Writing the classification of each point to the output file.
    if (rank == 0)
    {
        error = writeResult(classMap, lines, argv[6]);
        if (error != 0)
        {
            showFileError(error, argv[6]);
            // Don't call exit/error here, just print error and continue to allow all processes to finish gracefully
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

    free(local_data);
    free(local_classMap);
    free(local_pointsPerClass);
    free(local_auxCentroids);

    // END CLOCK*****************************************
    end_time = MPI_Wtime();
    printf("\n\nMemory deallocation: %f seconds\n", end_time - start_time);
    fflush(stdout);
    //***************************************************/
    MPI_Finalize();
    return 0;
}