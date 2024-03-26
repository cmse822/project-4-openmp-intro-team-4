#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include <cmath>

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

    int totalIterations = std::atoi(argv[2]); // Number of Runs
    int req_thread = std::atoi(argv[3]);      // # of threads requested
    const char* csvfile = argv[4]; // Name of the CSV file to

    int numtasks, rank;

    //Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    double sumRunTime = 0; // Add up run times for averaging
    double start_time, end_time, elapsed_time;

    // Check if file exists to avoid overwriting headers
    bool fileExists = std::ifstream(csvfile).good();

    start_time = MPI_Wtime(); //get start time using MPI function

    for (int iter = 0; iter < totalIterations; ++iter) {
        float** A = createMatrix(n, p);
        float** B = createMatrix(n, m);
        float** C = createMatrix(m, p);

        B = fillMatrixValues(B, n, m);
        C = fillMatrixValues(C, m, p);

        double startTime, endTime;

        get_walltime(&startTime); //use get_walltime so it works regardless 

        // If this is an OpenMP run, set the number of threads
        #ifdef _OPENMP
            omp_set_num_threads(req_thread);
        #endif

        // // Matrix-Matrix Multiplication
        /*MPI - distribute one of the rows of one of the input matrices 
        accross MPI ranks. */

        /*Distribute the "i" row of B accross MPI ranks. Therefore, 
        The i loop has dependency on the rank*/ 

        /*Determine the start and end row for MPI.*/
        int start_row = (n/numtasks)*rank;
        int end_row = start_row + ((n/numtasks)- 1);

        if (end_row >= n) {
            end_row = n-1;
        }

        // printf("\nstart row %d, end row %d, rank %d\n", start_row,end_row,rank);

        /*MPI Communication - Collect Results on a 
        Single Rank
        
        Notes: 
        - could only figure out how to send/recieve one row at a time
        */

        if (rank == 0) {
            #pragma omp parallel for collapse(2)
                for (int i=start_row; i<=end_row; i++) {
                    for (int j=0; j<m; j++) {
                        for (int k=0; k<p; k++) {
                            A[i][j] += B[i][k]*C[k][j];
                        }
                    }
                }
        } else {
            // Calculate number of rows in this rank
            int num_rows = end_row - start_row + 1;

            // Create an MPI_Request array with one request for each send operation
            MPI_Request requests[num_rows];

            #pragma omp parallel for
                for (int i=start_row; i<=end_row; i++) {
                    for (int j=0; j<m; j++) {
                        for (int k=0; k<p; k++) {
                            A[i][j] += B[i][k]*C[k][j];
                        }
                    }

                    // Only send if rank is not 0
                    MPI_Isend(&A[i][0], n, MPI_FLOAT, 0, i-start_row, MPI_COMM_WORLD, &requests[i-start_row]);
                }

            // Wait for all non-blocking operations to complete
            MPI_Waitall(num_rows, requests, MPI_STATUS_IGNORE);
        }
        
        get_walltime(&endTime);

        if (rank == 0) {
            int num_rows_task = n/numtasks;

            // Create an MPI_Request array with one request for each receive operation
            MPI_Request requests[n - num_rows_task];

            // Receive from all ranks other than zero and use threads to speed up the process
            #pragma omp parallel for
                for (int i = 1; i < numtasks; i++) {
                    if (i == numtasks - 1) {
                        num_rows_task = n - (num_rows_task * (numtasks - 1));
                    }

                    for (int row = 0; row < num_rows_task; row++) {
                        MPI_Irecv(&A[num_rows_task*i + row][0], n, MPI_FLOAT, i, row, MPI_COMM_WORLD, &requests[num_rows_task*(i - 1) + row]);
                    }
                }

            // Wait for all non-blocking operations to complete
            MPI_Waitall(n - num_rows_task, requests, MPI_STATUS_IGNORE);
        }

    // if rank is zero and its the last iteration
        if((rank == 0) && (iter == totalIterations -1)){
            printf("Entering serial check\n");

            // Allocate memory for the rows (array of pointers)
            bool** A_check = new bool*[n];
            bool all_true = true;
            float** A_serial = createMatrix(n, p);
            
            //Serial MMM
            for (int i = 0; i<n; i++) {
                for (int j = 0; j<m; j++) {
                    for (int k = 0; k<p; k++) {
                        A_serial[i][j] += B[i][k]*C[k][j];
                    }
                }
            }

            // Allocate memory for each row (array of boolean values)
            for (int i = 0; i < n; ++i) {
                A_check[i] = new bool[m];
            }     

            /*Check if the difference between the 
            parallel and serial versions is within 
            tolerance*/
            double diff = 0.0;
            for (int i = 0; i <n; i++) {
                for (int j = 0; j<m; j++) {
                     diff = A[i][j] - A_serial[i][j];

                    if (fabs(diff) < 1e-16) {
                        A_check[i][j] = true ;
                    }
                    else {
                        A_check[i][j] = false;
                        printf("!!!!Matrices are not equal!!!!\n");
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
            fileB.close();
            fileC.close();

            end_time = MPI_Wtime(); //get end time using MPI function
            elapsed_time = end_time - start_time;

            double meanRunTime = elapsed_time / totalIterations;

            /*If rank is zero, then run the serial calculation to compare 
                with the parallel version*/
                

            #ifdef _OPENMP

                int threads; // Declare an integer variable to store the maximum number of threads

                // Get the maximum number of threads
                threads = omp_get_max_threads();

            #else
            // NOTE: if threads = Serial, it is not an openMP run. it is a serial run. 
                const char* threads = "Serial";
            #endif

            printf("Writing Output Data\n");

            std::ofstream outputFile(csvfile, std::ios::app);
            // Write headers if file does not exist
            if (!fileExists) {
                outputFile << "Matrix Size,Iterations,OpenMP Threads,MPI Tasks,Average Runtime,Serial == Parallel?" << std::endl;
            }
            // Writing the data
            outputFile << n << ','<< totalIterations << "," << threads << "," << numtasks << ',' << meanRunTime << ','<< all_true <<std::endl;
            outputFile.close();


        }
        

            //     printf("\nCleaning up memory\n");
            //     // Cleanup memory
            //         deleteMatrix(A, n);
            //         deleteMatrix(B, n);
            //         deleteMatrix(C, m);

            //     printf("\nDone Cleaning up memory\n");
    }

    MPI_Finalize();
    return 0;
}
