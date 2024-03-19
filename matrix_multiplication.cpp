#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>
#include <omp.h>

using namespace std;

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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations> <n_threads>" << std::endl;
        return 1;
    }

    /*
    Program to perform Matrix-Matrix Multiplication
    A = BC
    */

    int n = std::atoi(argv[1]);     // # Rows of B
    int m = n;                      // # Columns of B & # Rows of C
    int p = n;                      // # Columns of C

    int req_thread = std::atoi(argv[3]);      // # of threads requested

    int totalIterations = std::atoi(argv[2]); // Number of Runs
    
    double sumRunTime = 0; // Add up run times for averaging

    // Check if file exists to avoid overwriting headers
    bool fileExists = std::ifstream("MatrixMultiplication.csv").good();

    std::ofstream outputFile("MatrixMultiplication.csv", std::ios::app);
    // Write headers if file does not exist
    if (!fileExists) {
        outputFile << "Matrix Size, Iterations, Threads, Average Runtime" << std::endl;
    }

    for (int iter = 0; iter < totalIterations; ++iter) {
        float** A = createMatrix(n, p);
        float** B = createMatrix(n, m);
        float** C = createMatrix(m, p);

        B = fillMatrixValues(B, n, m);
        C = fillMatrixValues(C, m, p);

        // double startTime, endTime; // Changed timing algorithm
        // get_walltime(&startTime);
        double startTime, endTime;

        get_walltime(&startTime); //use get_walltime so it works regardless 

    // If this is an OpenMP run, set the number of threads
        #ifdef _OPENMP
            omp_set_num_threads(req_thread);
        #endif

        // // Matrix-Matrix Multiplication
        #pragma omp parallel for collapse(2) // Collapse for loops
            for (int i = 0; i<n; i++) {
                for (int j = 0; j<m; j++) {
                    for (int k = 0; k<p; k++) {
                    A[i][j] += B[i][k]*C[k][j];
                    }
                }
            }

        get_walltime(&endTime);
    

                

        // Cleanup memory
        deleteMatrix(A, n);
        deleteMatrix(B, n);
        deleteMatrix(C, m);

        // Output runtime to console   
        double runTime = endTime - startTime;
        // cout << runTime << endl;

        sumRunTime += runTime;
    }

    double meanRunTime = sumRunTime / totalIterations;

    // std::ofstream outputFile("MatrixMultiplication.txt", std::ios::app); // Create MeanRunTime.txt file
    // outputFile << meanRunTime << std::endl; // Append content to the file
    // outputFile.close(); // Close the file

    #ifdef _OPENMP

        int threads; // Declare an integer variable to store the maximum number of threads

        // Get the maximum number of threads
        threads = omp_get_max_threads();

    #else
    // NOTE: if threads = Serial, it is not an openMP run. it is a serial run. 
        const char* threads = "Serial";
    #endif


    // Writing the data
    outputFile << n << ','<< totalIterations << "," << threads << "," << meanRunTime << std::endl;
    outputFile.close();

    
    // printf("hello");
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