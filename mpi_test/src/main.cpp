#include <gtest/gtest.h>

#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
#endif
#include <cuda_runtime.h>
#include <cstdlib>
int main(int argc, char** argv) {
#ifdef ALSVINN_USE_MPI
#ifdef DALSVINN_HAS_GPU_DIRECT
    setenv("MPICH_RDMA_ENABLED_CUDA", "1", 1);
    setenv("MV2_USE_CUDA", "1", 1);
#endif
    MPI_Init(NULL, NULL);
#ifdef DALSVINN_HAS_GPU_DIRECT

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* a;
    cudaMalloc(&a, sizeof(a));

    if (size > 1) {
        MPI_Request requestSend;
        MPI_Isend(a, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD,
            &requestSend);
        MPI_Request requestReceive;
        MPI_Irecv(a, 1, MPI_DOUBLE, (rank + 1) % size, 0, MPI_COMM_WORLD,
            &requestReceive);
        MPI_Wait(&requestSend, MPI_STATUS_IGNORE);
        MPI_Wait(&requestReceive, MPI_STATUS_IGNORE);

    }

#endif
#endif



    ::testing::InitGoogleTest(&argc, argv);

    auto exitValue = RUN_ALL_TESTS();
    MPI_Finalize();
    return exitValue;
}
