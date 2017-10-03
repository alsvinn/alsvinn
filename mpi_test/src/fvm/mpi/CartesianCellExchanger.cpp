#include <gtest/gtest.h>
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsfvm/volume/make_volume.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "do_serial.hpp"

using namespace alsfvm;

TEST(CartesianCellExchanger, Test1D) {
    auto mpiConfiguration = alsfvm::make_shared<mpi::Configuration>(MPI_COMM_WORLD);
    const int numberOfProcessors = mpiConfiguration->getNumberOfNodes();
    const int rank = mpiConfiguration->getNodeNumber();


    const int N = 16*numberOfProcessors;

    rvec3 lowerCorner = {0,0,0};
    rvec3 upperCorner = {1,0,0};

    const std::string platform = "cpu";
    const std::string equation = "burgers";

    const int ghostCells = 3;


    auto grid = alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, ivec3{N*numberOfProcessors,1,1},
                                          boundary::allPeriodic());


    auto volume = volume::makeConservedVolume(platform,
                                              equation,
    {N,1,1},
                                              ghostCells);


    mpi::domain::CartesianDecomposition decomposer(numberOfProcessors, 1, 1);

    auto information = decomposer.decompose(mpiConfiguration, *grid);

    auto newGrid = information->getGrid();
    auto newDimensions = newGrid->getDimensions();
    ASSERT_EQ(N, newDimensions.x);
    ASSERT_EQ(1, newDimensions.y);
    ASSERT_EQ(1, newDimensions.z);


    for(int side= 0; side < 6; ++side) {
        ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side));
    }



    for (int i = ghostCells; i < N + ghostCells; ++i) {
        (*volume->getScalarMemoryArea(0))[i] = i-ghostCells + N*rank;
    }
    const real magicValue = 42*numberOfProcessors+rank;
    for(int i = 0; i < ghostCells; ++i) {
        (*volume->getScalarMemoryArea(0))[i] = magicValue;
        (*volume->getScalarMemoryArea(0))[N+ghostCells + i] = magicValue;
    }


    for (int i = ghostCells; i < N + ghostCells; ++i) {
        auto value = (*volume->getScalarMemoryArea(0))[i];
        ASSERT_EQ(i-ghostCells + N*rank, value);
    }

    // Make sure max works
    real waveSpeed = 42*rank;
    real maxWaveSpeed = information->getCellExchanger()->adjustWaveSpeed(waveSpeed);

    ASSERT_EQ(42*(numberOfProcessors-1), maxWaveSpeed);


    information->getCellExchanger()->exchangeCells(*volume, *volume).waitForAll();




    if(numberOfProcessors==1) {
        return;
    }



    // left side
    for (int i = 0; i < ghostCells; ++i) {
        int index = i;
        auto value = (*volume->getScalarMemoryArea(0))[index];
        std::cout << value << std::endl;
        int expectedValue = N*rank - (ghostCells-i);
        if (rank == 0) {
            expectedValue = N*(numberOfProcessors) - (ghostCells-i);
        }

        EXPECT_EQ(expectedValue, value)
                << "Failed at left ghost index " << i << "  on processor " << rank;
    }

    // right side
    for (int i = 0; i < ghostCells; ++i) {
        int index = N+ghostCells+i;
        auto value = (*volume->getScalarMemoryArea(0))[index];
        std::cout << value << std::endl;
        int expectedValue = N*((rank+1)%numberOfProcessors) + (i);

        EXPECT_EQ(expectedValue, value)
                << "Failed at right ghost index " << i << "  on processor " << rank;
    }

#if 0 //debug output
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < N + 2*ghostCells; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }
        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        for (int i = 0; i < N + 2*ghostCells; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }
        std::cout << "_______________________________" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

}
