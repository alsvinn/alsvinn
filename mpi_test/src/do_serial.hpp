#pragma once
#define ALSVINN_DO_MPI_SERIAL(X, rank, numberOfProcessors) { \
    for(int cpu = 0; cpu < rank; ++cpu) { \
        MPI_Barrier(MPI_COMM_WORLD); \
        if (cpu == rank) { \
            X; \
        }\
        MPI_Barrier(MPI_COMM_WORLD); \
    } \
}
