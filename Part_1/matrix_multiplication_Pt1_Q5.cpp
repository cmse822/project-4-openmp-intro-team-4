/*******************************************************************
Team 4 - Project 4, Part 1, Q5:

This version runs the matrix multiplication both in serial mode 
and using openmp. It then checks if the resultant matrices are equal
within a specified tolerance (to account for machine precision). The 
serial result is compared to the parallel result for multiple thread
counts and matrix sizes. The results are stored in pt1_q5.csv


Notes:
  - found that A needed to be initialized with all zeros. 
********************************************************************/

#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include <iomanip> // Include this for std::setprecision


using namespace std;

float** createMatrix(int rows, int cols) {
    // "Double" Pointer: Create pointer to memory that stores rows of pointers
    float** matrix = new float*[rows];

    for (int i = 0; i < rows; i++) {  // Iterate over rows
        matrix[i] = new float[cols];  // Allocate memory for column at each row pointer

        // Initialize elements to zero
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0.0;
        }
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

    
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size> <iterations> <n_threads> <csv file name>" << std::endl;
        return 1;
    }

    /*
    Program to perform Matrix-Matrix Multiplication
    A = BC
    */

    int n = std::atoi(argv[1]);     // # Rows of B
    int m = n;                      // # Columns of B & # Rows of C
    int p = n;                      // # Columns of C

    // Allocate memory for the rows (array of pointers)
    bool** A_check = new bool*[n];
    bool all_true = true;

    int req_thread = std::atoi(argv[3]);      // # of threads requested

    int totalIterations = std::atoi(argv[2]); // Number of Runs

    const char* csvfile = argv[4]; // Name of the CSV file to

    // cout << csvfile << endl; 
    
    double sumRunTime = 0; // Add up run times for averaging

    // Check if file exists to avoid overwriting headers
    bool fileExists = std::ifstream(csvfile).good();

    std::ofstream outputFile(csvfile, std::ios::app);
    // Write headers if file does not exist
    if (!fileExists) {
        outputFile << "Matrix Size,Threads,Serial == Parallel Result?" << std::endl;
    }

    for (int iter = 0; iter < totalIterations; ++iter) {
        float** A = createMatrix(n, p);
        float** A_serial = createMatrix(n, p);
        float** B = createMatrix(n, m);
        float** C = createMatrix(m, p);

        B = fillMatrixValues(B, n, m);
        C = fillMatrixValues(C, m, p);

        double startTime, endTime;

        /*Serial Matrix-Matrix Multiplication
        This is used to verify that the solution does not 
        depend on the number of threads*/    
        for (int i = 0; i<n; i++) {
                for (int j = 0; j<m; j++) {
                    for (int k = 0; k<p; k++) {
                    A_serial[i][j] += B[i][k]*C[k][j];
                    }
                }
            }

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
    
    
        // Allocate memory for each row (array of boolean values)
        for (int i = 0; i < n; ++i) {
            A_check[i] = new bool[m];
        }     

        float diff = 0.0;
        for (int i = 0; i <n; i++) {
            for (int j = 0; j<m; j++) {
                double diff = A[i][j] - A_serial[i][j];

                if (abs(diff) < 1e-16) {
                    A_check[i][j] = true ;
                }
                else {
                    A_check[i][j] = false;
                    printf("%lf\n",diff); // Use %lf for double
                    printf("\nMatrices are not equal");
                    all_true = false; //Flag if one of the elements is not equal. 
                    break;
                    // return false;
                }
            if (all_true == false) {break;}
            

            }
        }

        //Write out matrices for debugging
        std::ofstream fileA("A.txt");
        std::ofstream fileA_serial("A_serial.txt");
        std::ofstream fileB("B.txt");
        std::ofstream fileC("C.txt");

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                fileA << A[i][j] << " ";
                fileA_serial << A_serial[i][j] << " ";
                fileB << B[i][j] << " ";
                fileC << C[i][j] << " ";
            }
            fileA << "\n";
            fileA_serial << "\n";
            fileB << "\n";
            fileC << "\n";
        }

        fileA.close();
        fileA_serial.close();

               
        
        // Cleanup memory
        deleteMatrix(A, n);
        deleteMatrix(A_serial, n);
        deleteMatrix(B, n);
        deleteMatrix(C, m);

        

        // Output runtime to console   
        double runTime = endTime - startTime;
        // cout << runTime << endl;

        sumRunTime += runTime;
    }

    double meanRunTime = sumRunTime / totalIterations;

    

    #ifdef _OPENMP

        int threads; // Declare an integer variable to store the maximum number of threads

        // Get the maximum number of threads
        threads = omp_get_max_threads();

    #else
    // NOTE: if threads = Serial, it is not an openMP run. it is a serial run. 
        const char* threads = "Serial";
    #endif

    // Writing the data
        outputFile << n << ',' << threads << "," << all_true << std::endl;
        outputFile.close();

  

    

    return 0;
}
