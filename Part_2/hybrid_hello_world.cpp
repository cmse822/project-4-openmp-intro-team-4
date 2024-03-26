#include <iostream>
#include <mpi.h>
#include <omp.h>

using namespace std;

int main(int argc, char* argv[]) 
{
    int numtasks, rank;

    cout << "Before initialization" << endl;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        cout << "This is thread " << id + 1 << " of " << num_threads << " in rank " << rank << endl;
    }
    
    MPI_Finalize();
    cout << "After initialize" << endl;

    return 0;
}