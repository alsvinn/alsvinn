#include <gtest/gtest.h>
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsfvm/volume/make_volume.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "do_serial.hpp"
#include "alsfvm/mpi/CartesianCellExchanger.hpp"

using namespace alsfvm;


TEST(CartesianCellExchanger3d, Test3D) {
     MPI_Barrier(MPI_COMM_WORLD);
    auto mpiConfiguration = alsfvm::make_shared<alsfvm::mpi::Configuration>(MPI_COMM_WORLD);
    const int numberOfProcessors = mpiConfiguration->getNumberOfProcesses();
    const int rank = mpiConfiguration->getRank();


    const int N = 8;

    if (numberOfProcessors < 8) {
        return;
    }


    rvec3 lowerCorner = {-3,3,-2};
    rvec3 upperCorner = {4,4,3};

    const std::string platform = "cpu";
    const std::string equation = "burgers";

    const int ghostCells = 3;


    int nz = numberOfProcessors/4;
    int ny = 2;
    int nx = 2;


    ASSERT_EQ(nx*ny*nz, numberOfProcessors);
    auto grid = alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, ivec3{N*nx,N*ny,N*nz},
                                          boundary::allPeriodic());


    auto volume = volume::makeConservedVolume(platform,
                                              equation,
        {N,N,N},
                                              ghostCells);


    alsfvm::mpi::domain::CartesianDecomposition decomposer(nx, ny, nz);

    auto information = decomposer.decompose(mpiConfiguration, *grid);

    auto newGrid = information->getGrid();
    auto newDimensions = newGrid->getDimensions();
    ASSERT_EQ(N, newDimensions.x);
    ASSERT_EQ(N, newDimensions.y);
    ASSERT_EQ(N, newDimensions.z);


    ivec3 numberOfProcessorsPerDirection = {nx, ny, nz};
    for(int side= 0; side < 6; ++side) {
        if (numberOfProcessorsPerDirection[side/2] != 1) {
            ASSERT_EQ(boundary::MPI_BC, newGrid->getBoundaryCondition(side));
        } else {
            ASSERT_EQ(boundary::PERIODIC, newGrid->getBoundaryCondition(side));
        }
    }


    const int M = N + 2*ghostCells;
    const real magicValue = N*N*N*42*numberOfProcessors+rank;
    for(int i = 0; i < M*M*M; ++i) {
        (*volume->getScalarMemoryArea(0))[i] = magicValue;
    }

    // Computes the x component of the rank
    auto xComponent = [&](int r) {
        return r%nx;
    };

    // computes the y component of the rank
    auto yComponent = [&](int r) {
        return (r/nx)%ny;
    };

    // computes the y component of the rank
    auto zComponent = [&](int r) {
        return r/(nx*ny);
    };

    auto computeValue = [&](int i, int j, int k, int r) {

        return (i-ghostCells + xComponent(r)*N)
                + (j-ghostCells + yComponent(r)*N)*N
                + (k-ghostCells + zComponent(r)*N)*N*N;
    };

    auto rankIndex = [&](int x, int y, int z) {
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

        if (z < 0) {
            z += nz;
        }
        if (z > nz - 1) {
            z -= nz;
        }


        return x + y*nx + z*ny*nx;
    };

    for (int k = ghostCells; k < N + ghostCells; ++k) {
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                (*volume->getScalarMemoryArea(0))[j*M+i+k*M*M] = computeValue(i,j, k, rank);
            }
        }
    }




    for(int k = ghostCells; k < N+ghostCells; ++k) {
        for (int i = ghostCells; i < N + ghostCells; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                auto value = (*volume->getScalarMemoryArea(0))[j*M+i+k*M*M];
                ASSERT_EQ(computeValue(i,j, k, rank), value);
            }
        }
    }

    // Make sure max works
    real waveSpeed = 42*rank;
    real maxWaveSpeed = information->getCellExchanger()->adjustWaveSpeed(waveSpeed);

    ASSERT_EQ(42*(numberOfProcessors-1), maxWaveSpeed);


    auto neighbours = alsfvm::dynamic_pointer_cast<alsfvm::mpi::CartesianCellExchanger>(information->getCellExchanger())->getNeighbours();

    for (int i = 0; i < 6; ++i) {
        ASSERT_LE(0, neighbours[i]);
        ASSERT_LT(neighbours[i], numberOfProcessors);
    }
    int xRank = xComponent(rank);
    int yRank = yComponent(rank);
    int zRank = zComponent(rank);


    auto globalPosition = newGrid->getGlobalPosition();
    ASSERT_EQ(globalPosition[0], xRank*N);
    ASSERT_EQ(globalPosition[1], yRank*N);
    ASSERT_EQ(globalPosition[2], zRank*N);

    ASSERT_EQ(newGrid->getCellLengths()[0], grid->getCellLengths()[0]);
    ASSERT_EQ(newGrid->getCellLengths()[1], grid->getCellLengths()[1]);
    ASSERT_EQ(newGrid->getCellLengths()[2], grid->getCellLengths()[2]);


    auto newLowerCorner = newGrid->getOrigin();
    ASSERT_DOUBLE_EQ(lowerCorner[0] + xRank*N*grid->getCellLengths()[0], newLowerCorner[0])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;

    ASSERT_DOUBLE_EQ(lowerCorner[1] + yRank*N*grid->getCellLengths()[1], newLowerCorner[1])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;
    ASSERT_DOUBLE_EQ(lowerCorner[2] + zRank*N*grid->getCellLengths()[2], newLowerCorner[2])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;

    auto newUpperCorner = newGrid->getTop();
    ASSERT_DOUBLE_EQ(lowerCorner[0] + (xRank+1)*N*grid->getCellLengths()[0], newUpperCorner[0])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;
    ASSERT_DOUBLE_EQ(lowerCorner[1] + (yRank+1)*N*grid->getCellLengths()[1], newUpperCorner[1])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;

    ASSERT_DOUBLE_EQ(lowerCorner[2] + (zRank+1)*N*grid->getCellLengths()[2], newUpperCorner[2])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;








    // Cell midpoint test
    auto newMidpoints = newGrid->getCellMidpoints();
    auto oldMidpoints = grid->getCellMidpoints();


    for (int z = 0; z < N; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int indexLocal = z*N*N+y*N+x;

                int indexGlobal = (zRank*N+z)*grid->getDimensions().x*grid->getDimensions().y
                        + (yRank*N+y)*grid->getDimensions().x + xRank*N + x;

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
                        <<"\n\tN                  = " << N
                          ;


            }
        }
    }






    ASSERT_EQ(rankIndex(xRank-1, yRank, zRank), neighbours[0])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;


    ASSERT_EQ(rankIndex(xRank+1, yRank,zRank), neighbours[1])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;




   ASSERT_EQ(rankIndex(xRank, yRank - 1,zRank), neighbours[2])
           << "Failed with"
           << "\n\trank              = " << rank
           << "\n\tnx                = " << nx
           << "\n\tny                = " << ny
           << "\n\txRank             = " << xRank
           << "\n\tyRank             = " << yRank
           << "\n\tumberOfProcessors = " << numberOfProcessors;



    ASSERT_EQ(rankIndex(xRank, yRank + 1, zRank), neighbours[3])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;





    ASSERT_EQ(rankIndex(xRank, yRank,zRank-1), neighbours[4])
            << "Failed with"
            << "\n\trank              = " << rank
            << "\n\tnx                = " << nx
            << "\n\tny                = " << ny
            << "\n\txRank             = " << xRank
            << "\n\tyRank             = " << yRank
            << "\n\tumberOfProcessors = " << numberOfProcessors;



     ASSERT_EQ(rankIndex(xRank, yRank, zRank+1), neighbours[5])
             << "Failed with"
             << "\n\trank              = " << rank
             << "\n\tnx                = " << nx
             << "\n\tny                = " << ny
             << "\n\txRank             = " << xRank
             << "\n\tyRank             = " << yRank
             << "\n\tumberOfProcessors = " << numberOfProcessors;

 MPI_Barrier(MPI_COMM_WORLD);


    information->getCellExchanger()->exchangeCells(*volume, *volume).waitForAll();


    if(numberOfProcessors==1) {
        return;
    }



    // left side
    for(int k = ghostCells; k < N + ghostCells; ++k) {
        for (int i = 0; i < ghostCells; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                int index = i + j*M+k*M*M;
                auto value = (*volume->getScalarMemoryArea(0))[index];

                int expectedValue = computeValue((i+N),j,k,rankIndex(xRank - 1, yRank,zRank));

                ASSERT_EQ(expectedValue, value)
                        << "Failed at left ghost index " << i << " and j = " << j << "  on processor " << rank;
            }
        }
    }

    // right side
    for(int k = ghostCells; k < N + ghostCells; ++k) {
        for (int i = N+ghostCells; i < M; ++i) {
            for (int j = ghostCells; j < N + ghostCells; ++j) {
                int index = i + j*M+k*M*M;
                auto value = (*volume->getScalarMemoryArea(0))[index];

                int expectedValue = computeValue((i-N),j,k,rankIndex(xRank + 1, yRank,zRank));

                ASSERT_EQ(expectedValue, value)
                        << "Failed at right ghost index " << i << " and j = " << j << "  on processor " << rank;
            }
        }
    }


    // bottom side
    for(int k = ghostCells; k < N + ghostCells; ++k) {
        for (int i = ghostCells; i < N+ghostCells; ++i) {
            for (int j = 0; j < ghostCells; ++j) {
                int index = i + j*M+k*M*M;
                auto value = (*volume->getScalarMemoryArea(0))[index];

                int expectedValue = computeValue(i, j+N,k, rankIndex(xRank, yRank-1, zRank));

                ASSERT_EQ(expectedValue, value)
                        << "Failed at left ghost index " << i << " and j = " << j << "  on processor " << rank;
            }
        }
    }

    // top side
    for(int k = ghostCells; k < N + ghostCells; ++k) {
        for (int i = ghostCells; i < N+ghostCells; ++i) {
            for (int j = N+ghostCells; j < M; ++j) {
                int index = i + j*M+k*M*M;
                auto value = (*volume->getScalarMemoryArea(0))[index];

                int expectedValue = computeValue(i, j-N,k, rankIndex(xRank, yRank+1, zRank));

                ASSERT_EQ(expectedValue, value)
                        << "Failed at left ghost index " << i << " and j = " << j << "  on processor " << rank;
            }
        }
    }

    // front side
    for(int k = 0; k < ghostCells; ++k) {
        for (int i = ghostCells; i < N+ghostCells; ++i) {
            for (int j = ghostCells; j < N+ghostCells; ++j) {
                int index = i + j*M+k*M*M;
                auto value = (*volume->getScalarMemoryArea(0))[index];

                int expectedValue = computeValue(i, j,k+N, rankIndex(xRank, yRank, zRank-1));

                ASSERT_EQ(expectedValue, value)
                        << "Failed at left ghost index " << i << " and j = " << j << "  on processor " << rank;
            }
        }
    }

    // back side
    for(int k = N+ghostCells; k < N+2*ghostCells; ++k) {
        for (int i = ghostCells; i < N+ghostCells; ++i) {
            for (int j = ghostCells; j < N+ghostCells; ++j) {
                int index = i + j*M+k*M*M;
                auto value = (*volume->getScalarMemoryArea(0))[index];

                int expectedValue = computeValue(i, j,k-N, rankIndex(xRank, yRank, zRank+1));

                ASSERT_EQ(expectedValue, value)
                        << "Failed at left ghost index " << i << " and j = " << j << "  on processor " << rank;
            }
        }
    }

}
