#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include <mpi.h>

using namespace std;

/*
CURRENT ISSUE:

[gerlac37@dev-amd20 project-4-openmp-intro-team-4]$ mpicxx -fopenmp -o hybrid_mmm hybrid_mmm.cpp && mpirun -np 4 ./hybrid_mmm 1000 10 4 results.csv 4
[dev-amd20:92809] *** Process received signal ***
[dev-amd20:92809] Signal: Segmentation fault (11)
[dev-amd20:92809] Signal code: Address not mapped (1)
[dev-amd20:92809] Failing at address: (nil)
[dev-amd20:92810] *** Process received signal ***
[dev-amd20:92808] *** Process received signal ***
--------------------------------------------------------------------------
mpirun noticed that process rank 1 with PID 0 on node dev-amd20 exited on signal 11 (Segmentation fault).
--------------------------------------------------------------------------
*/

float** createMatrix(int rows, int cols) {
    // "Double" Pointer: Create pointer to memory that stores rows of pointers
    float** matrix = new float*[rows];

    for (int i = 0; i < rows; i++) {  // Iterate over rows
        matrix[i] = new float[cols];  // Allocate memory for column at each row pointer
    }
    return matrix;
}

void deleteMatrix(float** matrix, int rows) {
    // rows = sizeof(matrix) / sizeof(matrix[0]) // Implement this line to take away the necesity of passing in rows
    for (int i = 0; i < rows; i++) {  // Iterate over rows
        delete[] matrix[i];  // Deallocate memory for column at each row pointer
    }
    delete[] matrix;  // Deallocate memory for original row points
}

float** fillMatrixValues(float** matrix, int rows, int cols) {
    for (int i = 0; i<rows; i++) {
        for (int j = 0; j<cols; j++) {
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }   
    }
    return matrix;
}

void printMatrixValues(float** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;    
    }
}

void get_walltime_(double* wcTime) {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);
}

void get_walltime(double* wcTime) {
    get_walltime_(wcTime);
}



int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 6) {
        if (world_rank == 0) { // Only rank 0 prints the error message
            cerr << "Usage: " << argv[0] << " <matrix_size> <iterations> <n_threads> <csv file name> <n_ranks>" << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort if the number of arguments is incorrect
    }

    /*
    Program to perform Matrix-Matrix Multiplication
    A = BC
    */

    int n = std::atoi(argv[1]);     // # Rows of B
    int m = n;                      // # Columns of B & # Rows of C
    int p = n;                      // # Columns of C

    int totalIterations = std::atoi(argv[2]);   // Number of Runs
    int req_thread = std::atoi(argv[3]);        // # of threads requested
    const char* csvfile = argv[4];              // Name of the CSV file to
    int mpi_ranks = atoi(argv[5]);              // Number of MPI Ranks

    double sumRunTime = 0; // Add up run times for averaging
    bool fileExists = false;
    std::ofstream outputFile;

    // Check if file exists to avoid overwriting headers
    if (world_rank == 0) {
        fileExists = std::ifstream(csvfile).good();
        outputFile.open(csvfile, std::ios::app);
        if (!fileExists) {     // Write headers if file does not exist
            outputFile << "Matrix Size, Iterations, Threads, MPI Ranks, Average Runtime" << std::endl;
        }
    }

    for (int iter = 0; iter < totalIterations; ++iter) {
        float** A = nullptr;
        if (world_rank == 0) {
            A = createMatrix(n, n); // A is only fully formed on the root process
        }
        float** B = createMatrix(n / mpi_ranks, n); // Divide B among processes
        float** C = createMatrix(n, n); // Full C is duplicated across all processes

        B = fillMatrixValues(B, n / mpi_ranks, n);
        C = fillMatrixValues(C, n, n);

        double startTime, endTime;
        MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks start timing at roughly the same moment
        get_walltime(&startTime);

        omp_set_num_threads(req_thread);
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n / mpi_ranks; i++) {
            for (int j = 0; j < n; j++) {
                for (int k = 0; k < n; k++) {
                    A[i][j] += B[i][k] * C[k][j]; // This part needs to be adjusted based on your MPI data distribution
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD); // Ensure all ranks finish before stopping the timer
        get_walltime(&endTime);

        // Gather the partial results from all processes to the root process
        // Note: You'll need to use MPI_Gather or similar to collect A's parts from all ranks

        if (world_rank == 0) {
            // Only root process performs the cleanup of A
            deleteMatrix(A, n);
        }
        deleteMatrix(B, n / mpi_ranks);
        deleteMatrix(C, n);

        double runTime = endTime - startTime;
        sumRunTime += runTime;
    }

    if (world_rank == 0) {
        double meanRunTime = sumRunTime / totalIterations;
        outputFile << n << ',' << totalIterations << "," << req_thread << "," << mpi_ranks << "," << meanRunTime << std::endl;
        outputFile.close();
    }

    MPI_Finalize();
    return 0;
}


/*

main() {
    int var1, var2, var3

    serial code
}

    #pragma omp parallel private(var1,var2) shared(var3)
    {
        Parallel region executed by all threads
        Other OpenMP Directives
        RunTime Library Calls
        All threads join master thread and disband
    }

    Resume serial code
*/