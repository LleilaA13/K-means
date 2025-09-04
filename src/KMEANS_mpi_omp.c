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
    int idx;
    for (int i = 0; i < K; i++)
    {
        idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
    }
}

/*
Function euclideanDistance: Euclidean distance (optimized with SIMD)
*/
float euclideanDistance(float *restrict point, float *restrict center, int samples)
{
    float dist = 0.0;

#pragma omp simd reduction(+ : dist)
    for (int i = 0; i < samples; i++)
    {
        // Use fmaf for better performance and accuracy
        dist = fmaf(point[i] - center[i], point[i] - center[i], dist);
    }
    // sqrt() is not necessary and increases execution time
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
    // Initialize MPI
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int nthreads = 4;
    if (argc == 9)                // 8 actual arguments + program name = 9
        nthreads = atoi(argv[8]); // argv[8] is the 8th argument (0-indexed)
    omp_set_num_threads(nthreads);
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
     * argv[8]: Number of threads to use.
     * */
    if ((argc != 9) && (argc != 8)) // 8 or 7 actual arguments + program name
    {
        fprintf(stderr, "EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr, "./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file] [Seed] [Optional: Number of threads]\n");
        fprintf(stderr, "Received %d arguments (argc=%d)\n", argc - 1, argc);
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Reading the input data
    // lines = number of points (rows); samples = number of dimensions per point (columns)
    int lines = 0, samples = 0;
    float *data = NULL;
    int error = 0;
    if (rank == 0)
    {
        error = readInput(argv[1], &lines, &samples);
        if (error != 0)
        {
            showFileError(error, argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    // Broadcast lines and samples to all ranks
    MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate and read data on rank 0
    if (rank == 0)
    {
        data = (float *)malloc(lines * samples * sizeof(float));
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
    }

    // Calculate data distribution parameters
    int local_lines = lines / size;
    int remainder = lines % size;
    if (rank < remainder)
    {
        local_lines++;
    }

    // Allocate local data for all processes
    float *local_data = (float *)malloc(local_lines * samples * sizeof(float));
    if (local_data == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Distribute data using Scatterv
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0)
    {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        // Calculate send counts and displacements
        for (int i = 0; i < size; i++)
        {
            int proc_lines = lines / size + (i < remainder ? 1 : 0);
            sendcounts[i] = proc_lines * samples;
            displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
        }
    }

    MPI_Scatterv(data, sendcounts, displs, MPI_FLOAT,
                 local_data, local_lines * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Clean up send arrays on rank 0
    if (rank == 0)
    {
        free(sendcounts);
        free(displs);
    }
    // Parameters
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    float *centroids = (float *)calloc(K * samples, sizeof(float));
    int *centroidPos = NULL;
    int *classMap = NULL;
    if (centroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    if (rank == 0)
    {
        centroidPos = (int *)calloc(K, sizeof(int));
        classMap = (int *)calloc(lines, sizeof(int));
        if (centroidPos == NULL || classMap == NULL)
        {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int seed = atoi(argv[7]);
        srand(seed);
        for (int i = 0; i < K; i++)
            centroidPos[i] = rand() % lines;
        initCentroids(data, centroids, centroidPos, samples, K);
    }
    // Broadcast centroids to all ranks - single MPI call, all data at once
    MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // END CLOCK*****************************************
    end_time = MPI_Wtime();
    printf("\nMemory allocation: %f seconds\n", end_time - start_time);
    fflush(stdout);
    //**************************************************
    // START CLOCK***************************************
    start_time = MPI_Wtime();
    //**************************************************
    char *outputMsg = (char *)calloc(10000, sizeof(char));

    int i, j; // Loop variables declared outside pragmas (like OMP version)
    int class;
    float dist, minDist;
    int it = 0;
    int changes = 0;
    float maxDist = 0.0; // Initialize to avoid warning

    // pointPerClass: number of points classified in each class
    // auxCentroids: mean of the points in each class
    float *pointsPerClass = (float *)malloc(K * sizeof(float)); // Use float for easier Allreduce packing
    float *auxCentroids = (float *)malloc(K * samples * sizeof(float));
    float *distCentroids = (float *)malloc(K * sizeof(float));

    if (pointsPerClass == NULL || auxCentroids == NULL || distCentroids == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Packed buffer for efficient single MPI_Allreduce (like in plain MPI version)
    size_t allgather_buffer_size = (1 + K + (K * samples));
    float *allgather_buffer = (float *)malloc(allgather_buffer_size * sizeof(float));

    /*
     * SIMPLE & EFFECTIVE HYBRID MPI+OpenMP STRATEGY:
     *
     * 1. MPI: Distribute data once, minimal communication per iteration
     * 2. OpenMP: Parallelize all computational loops with simple, proven patterns
     * 3. Focus on the 3 core parallelizable operations:
     *    - Distance calculation & assignment
     *    - Centroid accumulation
     *    - Centroid averaging
     */

    // Local variables for each process
    float *local_pointsPerClass = (float *)malloc(K * sizeof(float)); // Use float for consistency
    float *local_auxCentroids = (float *)malloc(K * samples * sizeof(float));
    int *local_classMap = (int *)calloc(local_lines, sizeof(int));

    if (local_pointsPerClass == NULL || local_auxCentroids == NULL || local_classMap == NULL || allgather_buffer == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    // Row pointers optimization for faster data access (same as OMP version)
    float **local_row_pointers = (float **)malloc(local_lines * sizeof(float *));
    if (local_row_pointers == NULL)
    {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

#pragma omp parallel for
    for (i = 0; i < local_lines; ++i)
    {
        local_row_pointers[i] = &local_data[i * samples];
    }

    do
    {
        it++;

        // STEP 1: Assign each point to nearest centroid (OpenMP parallelized)
        int local_changes = 0;

#pragma omp parallel for private(i, j, class, minDist, dist) shared(local_data, centroids, local_classMap, local_lines, samples, K) reduction(+ : local_changes) schedule(dynamic, 128)
        for (i = 0; i < local_lines; i++)
        {
            class = 1;
            minDist = FLT_MAX;

            // Cache-friendly: process all centroids for current point (using row pointers like OMP version)
            for (j = 0; j < K; j++)
            {
                dist = euclideanDistance(local_row_pointers[i], &centroids[j * samples], samples);
                if (dist < minDist)
                {
                    minDist = dist;
                    class = j + 1;
                }
            }

            // Count changes
            if (local_classMap[i] != class)
            {
                local_changes++;
            }
            local_classMap[i] = class;
        }

        // STEP 2: Accumulate points for each centroid (OpenMP parallelized with thread-local arrays)
        memset(local_pointsPerClass, 0, K * sizeof(float));
        memset(local_auxCentroids, 0, K * samples * sizeof(float));

#pragma omp parallel
        {
            // Thread-local accumulators
            float *thread_pointsPerClass = (float *)calloc(K, sizeof(float));
            float *thread_auxCentroids = (float *)calloc(K * samples, sizeof(float));

            // Each thread processes its portion of data with dynamic scheduling for better load balance
#pragma omp for private(i, j, class) schedule(dynamic, 64)
            for (i = 0; i < local_lines; i++)
            {
                class = local_classMap[i] - 1; // Convert to 0-based
                thread_pointsPerClass[class] += 1.0f;

                // Vectorize inner loop for better cache performance
                for (j = 0; j < samples; j++)
                {
                    thread_auxCentroids[class * samples + j] += local_data[i * samples + j];
                }
            }

            // OpenMP reduction: minimize critical section time
#pragma omp critical
            {
                // Combine results efficiently - unroll when possible
                for (i = 0; i < K; i++)
                {
                    local_pointsPerClass[i] += thread_pointsPerClass[i];
                }
                for (i = 0; i < K; i++)
                {
                    float *dest = &local_auxCentroids[i * samples];
                    float *src = &thread_auxCentroids[i * samples];
                    for (j = 0; j < samples; j++)
                    {
                        dest[j] += src[j];
                    }
                }
            }

            free(thread_pointsPerClass);
            free(thread_auxCentroids);
        }

        // STEP 3: MPI Reduction - combine results from all processes using packed buffer
        float float_changes = (float)local_changes;
        allgather_buffer[0] = float_changes;
        memcpy(&allgather_buffer[1], local_pointsPerClass, K * sizeof(float));
        memcpy(&allgather_buffer[1 + K], local_auxCentroids, K * samples * sizeof(float));

        MPI_Allreduce(MPI_IN_PLACE, allgather_buffer, allgather_buffer_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        changes = (int)allgather_buffer[0];
        memcpy(pointsPerClass, &allgather_buffer[1], K * sizeof(float));
        memcpy(auxCentroids, &allgather_buffer[1 + K], K * samples * sizeof(float));

        // Early termination check
        if (changes <= minChanges)
        {
            maxDist = 0.0;
            break;
        }

        // STEP 4: Compute new centroids (OpenMP parallelized)
#pragma omp parallel for private(i, j) collapse(2)
        for (i = 0; i < K; i++)
        {
            for (j = 0; j < samples; j++)
            {
                if (pointsPerClass[i] > 0.0f)
                    auxCentroids[i * samples + j] /= pointsPerClass[i];
                else
                    auxCentroids[i * samples + j] = centroids[i * samples + j];
            }
        }

        // STEP 5: Check centroid movement for convergence
        maxDist = FLT_MIN;
#pragma omp parallel for reduction(max : maxDist)
        for (i = 0; i < K; i++)
        {
            distCentroids[i] = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
            maxDist = distCentroids[i];
        }

        // Update centroids
        memcpy(centroids, auxCentroids, (K * samples * sizeof(float)));

        // Check convergence
        if (maxDist <= (maxThreshold * maxThreshold))
        {
            break;
        }

    } while (it < maxIterations);

    // FINAL MPI COMMUNICATION: Gather results for output
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *gather_displs = (int *)malloc(size * sizeof(int));

    MPI_Gather(&local_lines, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int total_lines = recvcounts[0];
        gather_displs[0] = 0;
        for (int i = 1; i < size; i++)
        {
            gather_displs[i] = gather_displs[i - 1] + recvcounts[i - 1];
            total_lines += recvcounts[i];
        }
        if (total_lines != lines)
        {
            fprintf(stderr, "Error: Mismatched line counts (expected %d, got %d)\n", lines, total_lines);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    if (rank == 0)
    {
        if (classMap == NULL)
        {
            classMap = (int *)calloc(lines, sizeof(int));
            if (classMap == NULL)
            {
                fprintf(stderr, "Memory allocation error.\n");
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }
    }
    MPI_Gatherv(local_classMap, local_lines, MPI_INT, classMap, recvcounts, gather_displs, MPI_INT, 0, MPI_COMM_WORLD);

    free(recvcounts);
    free(gather_displs);
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

    // Free memory - proper cleanup
    if (rank == 0 && data != NULL)
    {
        free(data);
    }
    if (rank == 0)
    {
        free(classMap);
        free(centroidPos);
    }
    free(centroids);
    free(distCentroids);
    free(pointsPerClass);
    free(auxCentroids);
    free(outputMsg);
    free(allgather_buffer);
    free(local_data);
    free(local_classMap);
    free(local_pointsPerClass);
    free(local_auxCentroids);
    free(local_row_pointers);

    // END CLOCK*****************************************
    end_time = MPI_Wtime();
    printf("\n\nMemory deallocation: %f seconds\n", end_time - start_time);
    fflush(stdout);
    //***************************************************/
    MPI_Finalize();
    return 0;
}