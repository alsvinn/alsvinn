/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtest/gtest.h>
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsfvm/volume/make_volume.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "do_serial.hpp"
#include "alsfvm/mpi/CartesianCellExchanger.hpp"
#include "alsfvm/mpi/Configuration.hpp"

using namespace alsfvm;
class CartesianCellExchangerTest :
    public ::testing::TestWithParam<std::string> {
public:
    CartesianCellExchangerTest()
        :
        platform(this->GetParam()) {
    }

    const std::string platform = "cpu";
};

TEST_P(CartesianCellExchangerTest, Test1D) {
    auto mpiConfiguration = alsfvm::make_shared<alsfvm::mpi::Configuration>
        (MPI_COMM_WORLD, platform);
    const int numberOfProcessors = mpiConfiguration->getNumberOfProcesses();
    const int rank = mpiConfiguration->getRank();


    const int N = 16 * numberOfProcessors;

    rvec3 lowerCorner = {0, 0, 0};
    rvec3 upperCorner = {1, 0, 0};

    const std::string equation = "burgers";

    const int ghostCells = 3;


    auto grid = alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, ivec3{N * numberOfProcessors, 1, 1},
            boundary::allPeriodic());


    auto volume = volume::makeConservedVolume(platform,
            equation,
    {N, 1, 1},
    ghostCells);


    alsfvm::mpi::domain::CartesianDecomposition decomposer(numberOfProcessors, 1,
        1);

    auto information = decomposer.decompose(mpiConfiguration, *grid);

    auto newGrid = information->getGrid();
    auto newDimensions = newGrid->getDimensions();
    ASSERT_EQ(N, newDimensions.x);
    ASSERT_EQ(1, newDimensions.y);
    ASSERT_EQ(1, newDimensions.z);


    auto globalPosition = newGrid->getGlobalPosition();

    ASSERT_EQ(globalPosition[0], rank * N);
    ASSERT_EQ(globalPosition[1], 0);
    ASSERT_EQ(globalPosition[2], 0);


    for (int side = 0; side < 6; ++side) {
        if (side < 2 && numberOfProcessors > 1) {
            ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side))
                    << "side = " << side;

        } else if (side < 2) {
            ASSERT_EQ(boundary::PERIODIC, newGrid->getBoundaryCondition(side))
                    << "side = " << side;;
        } else {
            ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side))
                    << "side = " << side;
        }
    }



    auto cpuVolume = volume->getCopyOnCPU();

    for (int i = ghostCells; i < N + ghostCells; ++i) {
        (*cpuVolume->getScalarMemoryArea(0))[i] = i - ghostCells + N * rank;
    }

    const real magicValue = 42 * numberOfProcessors + rank;

    for (int i = 0; i < ghostCells; ++i) {
        (*cpuVolume->getScalarMemoryArea(0))[i] = magicValue;
        (*cpuVolume->getScalarMemoryArea(0))[N + ghostCells + i] = magicValue;
    }

    if (platform != "cpu") {
        cpuVolume->copyTo(*volume);
    }

    for (int i = ghostCells; i < N + ghostCells; ++i) {
        auto value = (*cpuVolume->getScalarMemoryArea(0))[i];
        ASSERT_EQ(i - ghostCells + N * rank, value);
    }

    // Make sure max works
    real waveSpeed = 42 * rank;
    real maxWaveSpeed = information->getCellExchanger()->adjustWaveSpeed(waveSpeed);

    ASSERT_EQ(42 * (numberOfProcessors - 1), maxWaveSpeed);


    information->getCellExchanger()->exchangeCells(*volume, *volume).waitForAll();

    if (platform != "cpu") {
        volume->copyTo(*cpuVolume);
    }


    if (numberOfProcessors == 1) {
        return;
    }


#if 0 //debug output
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < N + 2 * ghostCells; ++i) {
            auto value = (*cpuVolume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        for (int i = 0; i < N + 2 * ghostCells; ++i) {
            auto value = (*cpuVolume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif


    // left side
    for (int i = 0; i < ghostCells; ++i) {
        int index = i;
        auto value = (*cpuVolume->getScalarMemoryArea(0))[index];

        int expectedValue = N * rank - (ghostCells - i);

        if (rank == 0) {
            expectedValue = N * (numberOfProcessors) - (ghostCells - i);
        }

        EXPECT_EQ(expectedValue, value)
                << "Failed at left ghost index " << i << "  on processor " << rank;
    }

    // right side
    for (int i = 0; i < ghostCells; ++i) {
        int index = N + ghostCells + i;
        auto value = (*cpuVolume->getScalarMemoryArea(0))[index];
        int expectedValue = N * ((rank + 1) % numberOfProcessors) + (i);

        EXPECT_EQ(expectedValue, value)
                << "Failed at right ghost index " << i << "  on processor " << rank;
    }

    // inner side
    for (int i = ghostCells; i < N + ghostCells; ++i) {
        auto value = (*cpuVolume->getScalarMemoryArea(0))[i];
        ASSERT_EQ(i - ghostCells + N * rank, value);
    }

}

TEST_P(CartesianCellExchangerTest, Test2D) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto mpiConfiguration = alsfvm::make_shared<alsfvm::mpi::Configuration>
        (MPI_COMM_WORLD, platform);
    const int numberOfProcessors = mpiConfiguration->getNumberOfProcesses();
    const int rank = mpiConfiguration->getRank();


    const int N = 8;//numberOfProcessors;

    if (numberOfProcessors < 4) {
        return;
    }


    rvec3 lowerCorner = {-3, 3, 0};
    rvec3 upperCorner = {4, 4, 0};


    const std::string equation = "burgers";

    const int ghostCells = 3;


    int nx = numberOfProcessors;
    int ny = 1;

    while (nx / ny > 2) {
        nx = nx / 2;
        ny = ny * 2;
    }

    ASSERT_EQ(nx * ny, numberOfProcessors);
    auto grid = alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, ivec3{N * nx, N * ny, 1},
            boundary::allPeriodic());


    auto volume = volume::makeConservedVolume(platform,
            equation,
    {N, N, 1},
    ghostCells);


    alsfvm::mpi::domain::CartesianDecomposition decomposer(nx, ny, 1);

    auto information = decomposer.decompose(mpiConfiguration, *grid);

    auto newGrid = information->getGrid();
    auto newDimensions = newGrid->getDimensions();
    ASSERT_EQ(N, newDimensions.x);
    ASSERT_EQ(N, newDimensions.y);
    ASSERT_EQ(1, newDimensions.z);

    for (int side = 0; side < 6; ++side) {
        ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side));
    }

    auto cpuVolume = volume->getCopyOnCPU();
    const int M = N + 2 * ghostCells;
    const real magicValue = N * N * 42 * numberOfProcessors + rank;

    for (int i = 0; i < M * M; ++i) {
        (*cpuVolume->getScalarMemoryArea(0))[i] = magicValue;
    }

    // Computes the x component of the rank
    auto xComponent = [&](int r) {
        return r % nx;
    };

    // computes the y component of the rank
    auto yComponent = [&](int r) {
        return r / nx;
    };

    auto computeValue = [&](int i, int j, int r) {

        return (i - ghostCells + xComponent(r) * N) + (j - ghostCells + yComponent(
                    r) * N) * N;
    };

    auto rankIndex = [&](int x, int y) {
        if (x < 0) {
            x += nx;
        }

        if (x > nx - 1) {
            x -= nx;
        }

        if (y < 0) {
            y += ny;
        }

        if (y > ny - 1) {
            y -= ny;
        }

        return x + y * nx;
    };

    for (int i = ghostCells; i < N + ghostCells; ++i) {
        for (int j = ghostCells; j < N + ghostCells; ++j) {
            (*cpuVolume->getScalarMemoryArea(0))[j * M + i] = computeValue(i, j, rank);
        }
    }




    for (int i = ghostCells; i < N + ghostCells; ++i) {
        for (int j = ghostCells; j < N + ghostCells; ++j) {
            auto value = (*cpuVolume->getScalarMemoryArea(0))[j * M + i];
            ASSERT_EQ(computeValue(i, j, rank), value);
        }
    }


    // Make sure max works
    real waveSpeed = 42 * rank;
    real maxWaveSpeed = information->getCellExchanger()->adjustWaveSpeed(waveSpeed);

    ASSERT_EQ(42 * (numberOfProcessors - 1), maxWaveSpeed);


    auto neighbours = information->getCellExchanger()->getNeighbours();

    for (int i = 0; i < 6; ++i) {
        ASSERT_LE(0, neighbours[i]);
        ASSERT_LT(neighbours[i], numberOfProcessors);
    }

    int xRank = xComponent(rank);
    int yRank = yComponent(rank);


    auto globalPosition = newGrid->getGlobalPosition();
    ASSERT_EQ(globalPosition[0], xRank * N);
    ASSERT_EQ(globalPosition[1], yRank * N);
    ASSERT_EQ(globalPosition[2], 0);

    ASSERT_EQ(newGrid->getCellLengths()[0], grid->getCellLengths()[0]);
    ASSERT_EQ(newGrid->getCellLengths()[1], grid->getCellLengths()[1]);
    ASSERT_EQ(newGrid->getCellLengths()[2], grid->getCellLengths()[2]);


    auto newLowerCorner = newGrid->getOrigin();
    ASSERT_DOUBLE_EQ(lowerCorner[0] + xRank * N * grid->getCellLengths()[0],
        newLowerCorner[0])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;

    ASSERT_DOUBLE_EQ(lowerCorner[1] + yRank * N * grid->getCellLengths()[1],
        newLowerCorner[1])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;
    ASSERT_DOUBLE_EQ(0, newLowerCorner[2]);

    auto newUpperCorner = newGrid->getTop();
    ASSERT_DOUBLE_EQ(lowerCorner[0] + (xRank + 1)*N * grid->getCellLengths()[0],
        newUpperCorner[0])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;
    ASSERT_DOUBLE_EQ(lowerCorner[1] + (yRank + 1)*N * grid->getCellLengths()[1],
        newUpperCorner[1])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;
    ASSERT_DOUBLE_EQ(0, newUpperCorner[2]);








    // Cell midpoint test
    auto newMidpoints = newGrid->getCellMidpoints();
    auto oldMidpoints = grid->getCellMidpoints();


    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int indexLocal = y * N + x;

            int indexGlobal = (yRank * N + y) * grid->getDimensions().x + xRank * N + x;

            ASSERT_EQ(oldMidpoints[indexGlobal], newMidpoints[indexLocal])
                    << "Failed with"
                        << "\n\trank              = " << rank
                        << "\n\tnx                = " << nx
                        << "\n\tny                = " << ny
                        << "\n\txRank             = " << xRank
                        << "\n\tyRank             = " << yRank
                        << "\n\tumberOfProcessors = " << numberOfProcessors
                        << "\n\tindexLocal        = " << indexLocal
                        << "\n\tindexGlobal       = " << indexGlobal
                        << "\n\tx                 = " << x
                        << "\n\ty                 = " << y
                        << "\n\tN                  = " << N
                        ;


        }
    }






    ASSERT_EQ(rankIndex(xRank - 1, yRank), neighbours[0])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;


    ASSERT_EQ(rankIndex(xRank + 1, yRank), neighbours[1])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;




    ASSERT_EQ(rankIndex(xRank, yRank - 1), neighbours[2])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;



    ASSERT_EQ(rankIndex(xRank, yRank + 1), neighbours[3])
            << "Failed with"
                << "\n\trank              = " << rank
                << "\n\tnx                = " << nx
                << "\n\tny                = " << ny
                << "\n\txRank             = " << xRank
                << "\n\tyRank             = " << yRank
                << "\n\tumberOfProcessors = " << numberOfProcessors;

    MPI_Barrier(MPI_COMM_WORLD);

    cpuVolume->copyTo(*volume);
    information->getCellExchanger()->exchangeCells(*volume, *volume).waitForAll();
    volume->copyTo(*cpuVolume);

    if (numberOfProcessors == 1) {
        return;
    }

#if 0 //debug output
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < M * M; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        for (int i = 0; i < M * M; ++i) {
            auto value = (*volume->getScalarMemoryArea(0))[i];
            std::cout << value << std::endl;
        }

        std::cout << "_______________________________" << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    // left side
    for (int i = 0; i < ghostCells; ++i) {
        for (int j = ghostCells; j < N + ghostCells; ++j) {
            int index = i + j * M;
            auto value = (*cpuVolume->getScalarMemoryArea(0))[index];

            int expectedValue = computeValue((i + N), j, rankIndex(xRank - 1, yRank));


            ASSERT_EQ(expectedValue, value)
                    << "Failed at left ghost index " << i << " and j = " << j << "  on processor "
                        << rank;
        }
    }

    // right side
    for (int i = N + ghostCells; i < M; ++i) {
        for (int j = ghostCells; j < N + ghostCells; ++j) {
            int index = i + j * M;
            auto value = (*cpuVolume->getScalarMemoryArea(0))[index];

            int expectedValue = computeValue((i - N), j, rankIndex(xRank + 1, yRank));

            ASSERT_EQ(expectedValue, value)
                    << "Failed at left ghost index " << i << " and j = " << j << "  on processor "
                        << rank;
        }
    }


    // bottom side
    for (int i = ghostCells; i < N + ghostCells; ++i) {
        for (int j = 0; j < ghostCells; ++j) {
            int index = i + j * M;
            auto value = (*cpuVolume->getScalarMemoryArea(0))[index];

            int expectedValue = computeValue(i, j + N, rankIndex(xRank, yRank - 1));

            ASSERT_EQ(expectedValue, value)
                    << "Failed at left ghost index " << i << " and j = " << j << "  on processor "
                        << rank;
        }
    }

    // top side
    for (int i = ghostCells; i < N + ghostCells; ++i) {
        for (int j = N + ghostCells; j < M; ++j) {
            int index = i + j * M;
            auto value = (*cpuVolume->getScalarMemoryArea(0))[index];

            int expectedValue = computeValue(i, j - N, rankIndex(xRank, yRank + 1));

            ASSERT_EQ(expectedValue, value)
                    << "Failed at left ghost index " << i << " and j = " << j << "  on processor "
                        << rank;
        }
    }

    // inner side
    for (int i = ghostCells; i < N + ghostCells; ++i) {
        for (int j = ghostCells; j < N + ghostCells; ++j) {
            auto value = (*cpuVolume->getScalarMemoryArea(0))[j * M + i];
            ASSERT_EQ(computeValue(i, j, rank), value);
        }
    }


}


INSTANTIATE_TEST_CASE_P(CartesianCellExchanger,
    CartesianCellExchangerTest,
    ::testing::Values("cpu"

        , "cuda"

    ));
