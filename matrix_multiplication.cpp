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
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations>" << std::endl;
        return 1;
    }

    /*
    Program to perform Matrix-Matrix Multiplication
    A = BC
    */

    int n = std::atoi(argv[1]); // # Rows of B
    int m = n; // # Columns of B & # Rows of C
    int p = n; // # Columns of C

    int totalIterations = std::atoi(argv[2]); // Number of Runs
    
    double sumRunTime = 0; // Add up run times for averaging

    for (int iter = 0; iter < totalIterations; ++iter) {
        float** A = createMatrix(n, p);
        float** B = createMatrix(n, m);
        float** C = createMatrix(m, p);

        B = fillMatrixValues(B, n, m);
        C = fillMatrixValues(C, m, p);

        double startTime, endTime;
        get_walltime(&startTime);

        // Matrix-Matrix Multiplication
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
        cout << runTime << endl;

        sumRunTime += runTime;
    }

    double meanRunTime = sumRunTime / totalIterations;

    std::ofstream outputFile("MeanRunTimes.txt", std::ios::app); // Create MeanRunTime.txt file
    outputFile << meanRunTime << std::endl; // Append content to the file
    outputFile.close(); // Close the file

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