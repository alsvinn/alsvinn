#include <gtest/gtest.h>

#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
#ifdef ALSVINN_USE_MPI
    MPI_Init(&argc, &argv);
#endif


    return RUN_ALL_TESTS();
}
