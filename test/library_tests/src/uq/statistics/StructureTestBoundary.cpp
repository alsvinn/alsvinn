#include <gtest/gtest.h>
#include "alsutils/config.hpp"

#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsuq/stats/StatisticsFactory.hpp"
#include "alsfvm/volume/make_volume.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/structure_common.hpp"

#include <random>
#include <map>

using namespace alsfvm;

namespace {
struct StructureParameters {

    const int p;
    const std::string platform;

    StructureParameters(
        const std::string& platform_,
        int p

    )
        :
        p(p),
#ifdef ALSVINN_HAVE_CUDA
        platform(platform_)
#else
        platform("cpu")
#endif
    {

    }

};

std::ostream& operator<<(std::ostream& os,
    const struct StructureParameters
    & parameters) {
    os << "\n{\n\tplatform = " << parameters.platform
        << "\n\tp = " << parameters.p << std::endl << "}" << std::endl;
    return os;
}


class TestWriter : public alsfvm::io::Writer {
public:

    TestWriter(alsfvm::volume::Volume& conservedVariablesSaved) :
        conservedVariablesSaved(conservedVariablesSaved) {

    }


    static std::shared_ptr<alsfvm::io::Writer> makeInstance(
        alsfvm::volume::Volume& conservedVariablesSaved) {
        std::shared_ptr<alsfvm::io::Writer> pointer;
        pointer.reset(new TestWriter(conservedVariablesSaved));
        return pointer;
    }



    alsfvm::volume::Volume& conservedVariablesSaved;

    void write(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::grid::Grid&,
        const alsfvm::simulator::TimestepInformation&) override  {

        conservedVariables.copyTo(conservedVariablesSaved);
    }
};


class StructureTestBoundary : public ::testing::TestWithParam
    <StructureParameters> {
public:

    const std::string platform;
    const int p;

    const int ghostCells = 3;
    const alsfvm::ivec3 innerSize = {16, 1, 1};
    const std::string equation = "euler3";


    const int numberOfH = 8;

    std::shared_ptr<alsuq::mpi::Configuration> mpiConfiguration;

    std::shared_ptr<alsfvm::DeviceConfiguration> deviceConfigurationCPU;

    std::shared_ptr<alsfvm::memory::MemoryFactory> memoryFactory;


    std::shared_ptr<alsfvm::volume::Volume> volumeConserved;
    std::shared_ptr<alsfvm::volume::Volume> conservedOutput;

    std::shared_ptr<alsfvm::io::Writer> writer;


    alsuq::stats::StatisticsFactory statisticsFactory;

    std::shared_ptr<alsuq::stats::Statistics> structure;

    alsfvm::grid::Grid gridPeriodic{alsfvm::rvec3{0, 0, 0},
               alsfvm::rvec3{1, 1, 1},
               innerSize, alsfvm::boundary::allPeriodic()};


    alsfvm::grid::Grid gridNeumann{alsfvm::rvec3{0, 0, 0},
               alsfvm::rvec3{1, 1, 1},
               innerSize, alsfvm::boundary::allNeumann()};

    alsfvm::simulator::TimestepInformation timestepInformation;



    StructureTestBoundary() :
        platform(GetParam().platform),
        p(GetParam().p),
        mpiConfiguration(std::make_shared<alsuq::mpi::Configuration>
            (MPI_COMM_WORLD, platform)),
        deviceConfigurationCPU(std::make_shared<alsfvm::DeviceConfiguration>(platform)),
        memoryFactory(std::make_shared<alsfvm::memory::MemoryFactory>
            (deviceConfigurationCPU)),
        volumeConserved(alsfvm::volume::makeConservedVolume(platform, equation,
                innerSize,
                ghostCells)),
        conservedOutput(alsfvm::volume::makeConservedVolume(platform, equation, {
        numberOfH, 1, 1
    },
    0)), writer(TestWriter::makeInstance(
            *conservedOutput))


    {

        volumeConserved->makeZero();
        conservedOutput->makeZero();


        resetStructure();
    }

    void resetGrid(ivec3 size) {
        gridPeriodic = grid::Grid{alsfvm::rvec3{0, 0, 0},
            alsfvm::rvec3{1, 1, 1},
            size, alsfvm::boundary::allPeriodic()};


        gridNeumann = grid::Grid{alsfvm::rvec3{0, 0, 0},
            alsfvm::rvec3{1, 1, 1},
            size, alsfvm::boundary::allNeumann()};

        volumeConserved = alsfvm::volume::makeConservedVolume(platform, equation,
                size,
                ghostCells);

        volumeConserved->makeZero();
    }

    void resetStructure() {
        boost::property_tree::ptree ptreeParameters;
        ptreeParameters.put("p", p);
        ptreeParameters.put("numberOfH", numberOfH);


        alsuq::stats::StatisticsParameters parameters(ptreeParameters);
        parameters.setMpiConfiguration(mpiConfiguration);
        parameters.setNumberOfSamples(1);
        structure = statisticsFactory.makeStatistics(platform,
                "structure_cube",
                parameters);

        for (auto statsName :
            structure->getStatisticsNames()) {

            structure->addWriter(statsName, writer);
        }
    }

};
}


TEST_P(StructureTestBoundary, TestConstant) {

    for (auto& grid : {
            gridNeumann, gridPeriodic
        }) {

        structure->write(*volumeConserved, grid,
            timestepInformation);

        structure->combineStatistics();
        structure->finalizeStatistics();
        structure->writeStatistics(grid);

        // Now check that everything is zero
        auto copyOnCPU = conservedOutput->getCopyOnCPU();

        alsfvm::volume::for_each_internal_volume_index(*copyOnCPU, [&](size_t index) {
            for (size_t var = 0; var < copyOnCPU->getNumberOfVariables(); ++var) {
                ASSERT_EQ(0.0, copyOnCPU->getScalarMemoryArea(var)->getPointer()[index]);
            }
        });
    }

}

TEST_P(StructureTestBoundary, TestLinear) {

    for (auto& grid : {
            gridNeumann, gridPeriodic
        }) {


        volumeConserved->makeZero();

        // Write new
        auto inputCopyOnCPU = volumeConserved->getCopyOnCPU();

        alsfvm::volume::for_each_midpoint(*inputCopyOnCPU, grid, [&](real, real,
                real,
        size_t index) {
            for (size_t var = 0; var < inputCopyOnCPU->getNumberOfVariables(); ++var) {
                inputCopyOnCPU->getScalarMemoryArea(var)->getPointer()[index] = int(
                        index) - int(
                        ghostCells);
            }
        });

        inputCopyOnCPU->copyTo(*volumeConserved);

        resetStructure();
        conservedOutput->makeZero();


        structure->write(*volumeConserved, grid,
            timestepInformation);

        structure->combineStatistics();
        structure->finalizeStatistics();
        structure->writeStatistics(grid);


        {
            // Now check that everything is set to the correct value
            auto copyOnCPU = conservedOutput->getCopyOnCPU();


            for (int h = 0; h < numberOfH; ++h) {
                for (size_t var = 0; var < copyOnCPU->getNumberOfVariables(); ++var) {

                    double correctValue = 0.0;

                    const int N = grid.getDimensions().x;

                    for (int i = 0; i < N; ++i) {
                        for (auto j : {
                                -h, h
                                }) {

                            ASSERT_EQ(i, inputCopyOnCPU->getScalarMemoryArea(var)->getPointer()[i +
                                      ghostCells]);

                            if (i + j >= 0 && i + j < N ) {
                                correctValue += std::pow(std::abs(-j), p);
                            } else if (i + j < 0) {
                                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                                    correctValue += std::pow(std::abs(i - (N + (i + j))), p);
                                } else {
                                    correctValue += std::pow(std::abs(i), p);
                                }
                            } else if (i + j >= N) {
                                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                                    correctValue += std::pow(std::abs(i - (i + j - N)), p);
                                } else {
                                    correctValue += std::pow(std::abs(i - (N - 1)), p);
                                }
                            }
                        }
                    }

                    correctValue /= N;

                    const std::string boundaryName = grid.getBoundaryCondition(
                            0) == boundary::PERIODIC ? "periodic" : "neumann";

                    if ((platform == "cpu" && p < 7) || p < 6) {
                        ASSERT_DOUBLE_EQ(correctValue, copyOnCPU->getScalarMemoryArea(
                                var)->getPointer()[h]) <<
                                    "Wrong structure function with boundary condition " << boundaryName
                                    << ", variable " << copyOnCPU->getName(var) << ", h = " << h;
                    } else {
                        // Due to round off errors in summation (CUDA is actually MORE accurate here),
                        // we need to show some lenience
                        ASSERT_NEAR(correctValue, copyOnCPU->getScalarMemoryArea(
                                var)->getPointer()[h], 1e-8 * correctValue) <<
                                    "Wrong structure function with boundary condition " << boundaryName
                                    << ", variable " << copyOnCPU->getName(var) << ", h = " << h;
                    }

                }
            }
        }
    }

}


TEST_P(StructureTestBoundary, TestLinear2D) {

    resetGrid({16, 16, 1});

    for (auto& grid : {
            gridNeumann, gridPeriodic
        }) {

        const int nx = grid.getDimensions().x;
        const int ny = grid.getDimensions().y;
        const int nz = grid.getDimensions().z;

        volumeConserved->makeZero();

        // Write new
        auto inputCopyOnCPU = volumeConserved->getCopyOnCPU();


        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                const int index = i + ghostCells + (j + ghostCells) * (nx + 2 * ghostCells);

                for (size_t var = 0; var < inputCopyOnCPU->getNumberOfVariables(); ++var) {
                    inputCopyOnCPU->getScalarMemoryArea(var)->getPointer()[index] = i + nx * j;
                }
            }
        }


        inputCopyOnCPU->copyTo(*volumeConserved);
        // now we check that the indexing is done correctly:
        // (note, we only do this on the CPU, but the same function is called on the GPU as well)
        auto coordinates = [&] (int index) {
            int i = index % nx;
            int j = (index / nx) % ny;
            int k = 1;

            return std::array<int, 3>({i, j, k});


        };

        if (platform == "cpu") {
            for (int h = 1; h < 4; ++h) {
                for (int k = h; k < nz - h; ++k) {
                    for (int j = h; j < ny - h; ++j) {
                        for (int i = h; i < nx - h; ++i) {


                            std::map<int, int> neighoursVisited;
                            int totalCount = 0;

                            auto testFunction = [&](double u, double v) {

                                int indexSource = int(u);
                                ASSERT_EQ(indexSource, u);
                                auto coordinatesSource = coordinates(indexSource);
                                auto coordinateTrue =  std::array<int, 3>({i, j, k});

                                for (size_t component = 0; component < 3u; ++component) {
                                    ASSERT_EQ(coordinatesSource[component], coordinateTrue[component]);
                                }

                                int indexTarget = int(v);
                                ASSERT_EQ(indexTarget, v);

                                ASSERT_NE(indexTarget, indexSource);
                                auto coordinatesTarget = coordinates(indexTarget);

                                for (size_t component = 0; component < 3u; ++component) {
                                    ASSERT_GT(coordinatesTarget[component], coordinatesSource[component] - (h + 1));
                                    ASSERT_LT(coordinatesTarget[component], coordinatesSource[component] + (h + 1));

                                }


                                EXPECT_EQ(neighoursVisited[indexTarget], 0)
                                        << "Repeated index: " << indexTarget
                                            << "\n\tsourceCoordinate = " << coordinatesSource[0] << ", " <<
                                            coordinatesSource[1] << ", " << coordinatesSource[2]
                                            << "\n\ttargetCoordinate = " << coordinatesTarget[0] << ", " <<
                                            coordinatesTarget[1] << ", " << coordinatesTarget[2]
                                            << "\n\trepeated: " << neighoursVisited[indexTarget]
                                            << "\nh: " << h << std::endl;

                                neighoursVisited[indexTarget]++;

                                totalCount += 1;


                            };

                            if (grid.getBoundaryCondition(0) == boundary::NEUMANN ) {

                                alsfvm::functional::forEachPointInComputeStructureCube<boundary::NEUMANN>
                                (testFunction
                                    ,

                                    std::dynamic_pointer_cast<const volume::Volume>
                                    (inputCopyOnCPU)->getScalarMemoryArea(0)->getView(),
                                    i, j, k, h, nx, ny, nz,
                                    3, 3, 3, 3);


                            } else if (grid.getBoundaryCondition(0) == boundary::PERIODIC ) {
                                alsfvm::functional::forEachPointInComputeStructureCube<boundary::PERIODIC>
                                (testFunction,

                                    std::dynamic_pointer_cast<const volume::Volume>
                                    (inputCopyOnCPU)->getScalarMemoryArea(0)->getView(),
                                    i, j, k, h, nx, ny, nz,
                                    3, 3, 3, 3);
                            }

                            int expectedTotalCount = 2 * ((2 * h + 1) * (2 * h + 1)) + (2 * h - 1) * (2 *
                                    (2 * h + 1) + 2 * (2 * h - 1));

                            ASSERT_EQ(totalCount, expectedTotalCount) << "Wrong number visited at "
                                << "\n\t" << i << " " << j << " " << k <<
                                "\n\th: " << h << std::endl;
                        }
                    }
                }
            }
        }

        resetStructure();
        conservedOutput->makeZero();


        structure->write(*volumeConserved, grid,
            timestepInformation);

        structure->combineStatistics();
        structure->finalizeStatistics();
        structure->writeStatistics(grid);

        auto boundaryIndex = [&](int n, int i, int offset, int boundaryValueLeft,
        int boundaryValueRight, int factor) {
            int offsetValue = -factor * offset;

            if (i + offset < 0) {
                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                    offsetValue = factor * (i - (n + (i + offset)));
                } else {
                    offsetValue = factor * i - boundaryValueLeft;
                }
            } else if (i + offset >= n) {
                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                    offsetValue = factor * (i - (i + offset - n));
                } else {
                    offsetValue = factor * i - (boundaryValueRight);
                }
            }

            return offsetValue;

        };


        {
            // Now check that everything is set to the correct value
            auto copyOnCPU = conservedOutput->getCopyOnCPU();


            for (int h = 1; h < numberOfH; ++h) {
                for (size_t var = 0; var < copyOnCPU->getNumberOfVariables(); ++var) {

                    double correctValue = 0.0;



                    for (int j = 0; j < ny; ++j) {
                        for (int i = 0; i < nx; ++i) {

                            ASSERT_NEAR(i + j * (nx), inputCopyOnCPU->getScalarMemoryArea(
                                    var)->getPointer()[i + ghostCells + (j + ghostCells) * (nx +
                                          2 * ghostCells)], 1e-300)
                                    << "Wrong input array at " << i << ", " << j;

                            auto updateCorrectValue = [&](int x, int y) {
                                //printf("i = %02d, j = %02d, x = %02d, y = %02d\n", i, j, x, y);
                                double xOffsetValue = boundaryIndex(nx, i, x, 0, (nx - 1), 1);

                                double yOffsetValue = boundaryIndex(ny, j, y, 0, (ny - 1) * nx, nx);




                                // The two tests below (in the if/else) is simply a test for the unit test...
                                // setting this up correctly was really difficult.
                                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                                    int iPlusH = i + x;
                                    int jPlusH = j + y;

                                    while (iPlusH < 0) {
                                        iPlusH += nx;
                                    }


                                    while (jPlusH < 0) {
                                        jPlusH += ny;
                                    }

                                    iPlusH %= nx;
                                    jPlusH %= ny;

                                    int indexNewH = (iPlusH + ghostCells) + (jPlusH + ghostCells) *
                                        (nx + 2 * ghostCells);
                                    int indexNew = (i + ghostCells) + (j + ghostCells) * (nx + 2 * ghostCells);

                                    auto u = inputCopyOnCPU->getScalarMemoryArea(
                                            var)->getPointer()[indexNew];


                                    auto uH = inputCopyOnCPU->getScalarMemoryArea(
                                            var)->getPointer()[indexNewH];


                                    ASSERT_DOUBLE_EQ(u - uH, xOffsetValue + yOffsetValue);

                                } else {
                                    int iPlusH = i + x;
                                    int jPlusH = j + y;

                                    if (iPlusH < 0) {
                                        iPlusH = 0;
                                    } else if (iPlusH >= nx) {
                                        iPlusH = nx - 1;
                                    }


                                    if (jPlusH < 0) {
                                        jPlusH = 0;
                                    } else if (jPlusH >= nx) {
                                        jPlusH = nx - 1;
                                    }

                                    int indexNewH = (iPlusH + ghostCells) + (jPlusH + ghostCells) *
                                        (nx + 2 * ghostCells);
                                    int indexNew = (i + ghostCells) + (j + ghostCells) * (nx + 2 * ghostCells);

                                    auto u = inputCopyOnCPU->getScalarMemoryArea(
                                            var)->getPointer()[indexNew];


                                    auto uH = inputCopyOnCPU->getScalarMemoryArea(
                                            var)->getPointer()[indexNewH];

                                    ASSERT_DOUBLE_EQ(u - uH, xOffsetValue + yOffsetValue);

#ifndef __CUDA_ARCH__

                                    if (var == 0) {
                                        std::ofstream outputFile("dumped_test.txt", std::ios::app);
                                        outputFile << h << " " << i << " " << j << " " << x << " " << y << " " << u <<
                                            " " <<  uH << std::endl;
                                    }

#endif

                                }

                                correctValue += std::pow(std::abs(xOffsetValue + yOffsetValue), p);
                            };

                            for (auto x : {
                                    -h, h
                                    }) {

                                for (int y = -h; y < h + 1; ++y) {
                                    // x side
                                    updateCorrectValue(x, y);

                                    // y side
                                    if (y > -h && y < h) {
                                        updateCorrectValue(y, x);
                                    }
                                }
                            }
                        }
                    }

                    correctValue /= (nx * ny);

                    const std::string boundaryName = grid.getBoundaryCondition(
                            0) == boundary::PERIODIC ? "periodic" : "neumann";

                    if ((platform == "cpu" && p < 7) || p < 6) {
                        ASSERT_DOUBLE_EQ(correctValue, copyOnCPU->getScalarMemoryArea(
                                var)->getPointer()[h]) <<
                                    "Wrong structure function with boundary condition " << boundaryName
                                    << ", variable " << copyOnCPU->getName(var) << ", h = " << h;
                    } else {
                        // Due to round off errors in summation (CUDA is actually MORE accurate here),
                        // we need to show some lenience
                        ASSERT_NEAR(correctValue, copyOnCPU->getScalarMemoryArea(
                                var)->getPointer()[h], 1e-8 * correctValue) <<
                                    "Wrong structure function with boundary condition " << boundaryName
                                    << ", variable " << copyOnCPU->getName(var) << ", h = " << h;
                    }

                }
            }
        }
    }

}


TEST_P(StructureTestBoundary, TestLinear3D) {

    resetGrid({16, 16, 16});

    for (auto& grid : {
            gridNeumann, gridPeriodic
        }) {


        const int nx = grid.getDimensions().x;
        const int ny = grid.getDimensions().y;
        const int nz = grid.getDimensions().z;

        volumeConserved->makeZero();

        // Write new
        auto inputCopyOnCPU = volumeConserved->getCopyOnCPU();


        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {

                    const int index = i + ghostCells + (j + ghostCells) * (nx + 2 * ghostCells) +
                        (k + ghostCells) * (nx + 2 * ghostCells) * (ny + 2 * ghostCells);

                    for (size_t var = 0; var < inputCopyOnCPU->getNumberOfVariables(); ++var) {
                        inputCopyOnCPU->getScalarMemoryArea(var)->getPointer()[index] = i + nx * j + nx
                            * ny * k;
                    }
                }
            }
        }


        inputCopyOnCPU->copyTo(*volumeConserved);

        // now we check that the indexing is done correctly:
        // (note, we only do this on the CPU, but the same function is called on the GPU as well)
        auto coordinates = [&] (int index) {
            int i = index % nx;
            int j = (index / nx) % ny;
            int k = (index / (nx * ny));

            return std::array<int, 3>({i, j, k});


        };

        if (platform == "cpu") {
            for (int h = 1; h < 4; ++h) {
                for (int k = h; k < nz - h; ++k) {
                    for (int j = h; j < ny - h; ++j) {
                        for (int i = h; i < nx - h; ++i) {


                            std::map<int, int> neighoursVisited;
                            int totalCount = 0;

                            auto testFunction = [&](double u, double v) {

                                int indexSource = int(u);
                                ASSERT_EQ(indexSource, u);
                                auto coordinatesSource = coordinates(indexSource);
                                auto coordinateTrue =  std::array<int, 3>({i, j, k});

                                for (size_t component = 0; component < 3u; ++component) {
                                    ASSERT_EQ(coordinatesSource[component], coordinateTrue[component]);
                                }

                                int indexTarget = int(v);
                                ASSERT_EQ(indexTarget, v);

                                ASSERT_NE(indexTarget, indexSource);
                                auto coordinatesTarget = coordinates(indexTarget);

                                for (size_t component = 0; component < 3u; ++component) {
                                    ASSERT_GT(coordinatesTarget[component], coordinatesSource[component] - (h + 1));
                                    ASSERT_LT(coordinatesTarget[component], coordinatesSource[component] + (h + 1));

                                }


                                EXPECT_EQ(neighoursVisited[indexTarget], 0)
                                        << "Repeated index: " << indexTarget
                                            << "\n\tsourceCoordinate = " << coordinatesSource[0] << ", " <<
                                            coordinatesSource[1] << ", " << coordinatesSource[2]
                                            << "\n\ttargetCoordinate = " << coordinatesTarget[0] << ", " <<
                                            coordinatesTarget[1] << ", " << coordinatesTarget[2]
                                            << "\n\trepeated: " << neighoursVisited[indexTarget]
                                            << "\nh: " << h << std::endl;

                                neighoursVisited[indexTarget]++;

                                totalCount += 1;


                            };

                            if (grid.getBoundaryCondition(0) == boundary::NEUMANN ) {

                                alsfvm::functional::forEachPointInComputeStructureCube<boundary::NEUMANN>
                                (testFunction
                                    ,

                                    std::dynamic_pointer_cast<const volume::Volume>
                                    (inputCopyOnCPU)->getScalarMemoryArea(0)->getView(),
                                    i, j, k, h, nx, ny, nz,
                                    3, 3, 3, 3);


                            } else if (grid.getBoundaryCondition(0) == boundary::PERIODIC ) {
                                alsfvm::functional::forEachPointInComputeStructureCube<boundary::PERIODIC>
                                (testFunction,

                                    std::dynamic_pointer_cast<const volume::Volume>
                                    (inputCopyOnCPU)->getScalarMemoryArea(0)->getView(),
                                    i, j, k, h, nx, ny, nz,
                                    3, 3, 3, 3);
                            }

                            int expectedTotalCount = 2 * ((2 * h + 1) * (2 * h + 1)) + (2 * h - 1) * (2 *
                                    (2 * h + 1) + 2 * (2 * h - 1));

                            ASSERT_EQ(totalCount, expectedTotalCount) << "Wrong number visited at "
                                << "\n\t" << i << " " << j << " " << k <<
                                "\n\th: " << h << std::endl;
                        }
                    }
                }
            }
        }

        resetStructure();
        conservedOutput->makeZero();


        structure->write(*volumeConserved, grid,
            timestepInformation);

        structure->combineStatistics();
        structure->finalizeStatistics();
        structure->writeStatistics(grid);

        auto boundaryIndex = [&](int n, int i, int offset, int boundaryValueLeft,
        int boundaryValueRight, int factor) {
            int offsetValue = -factor * offset;

            if (i + offset < 0) {
                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                    offsetValue = factor * (i - (n + (i + offset)));
                } else {
                    offsetValue = factor * i - boundaryValueLeft;
                }
            } else if (i + offset >= n) {
                if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                    offsetValue = factor * (i - (i + offset - n));
                } else {
                    offsetValue = factor * i - (boundaryValueRight);
                }
            }

            return offsetValue;

        };


        {
            // Now check that everything is set to the correct value
            auto copyOnCPU = conservedOutput->getCopyOnCPU();


            for (int h = 1; h < numberOfH; ++h) {
                for (size_t var = 0; var < copyOnCPU->getNumberOfVariables(); ++var) {

                    double correctValue = 0.0;


                    for (int k = 0; k < nz; ++k) {
                        for (int j = 0; j < ny; ++j) {
                            for (int i = 0; i < nx; ++i) {


                                ASSERT_NEAR(i + j * (nx) + k * (nx * ny), inputCopyOnCPU->getScalarMemoryArea(
                                        var)->getPointer()[i + ghostCells + (j + ghostCells) * (nx +
                                              2 * ghostCells) + (k + ghostCells) * (nx +
                                              2 * ghostCells) * (ny +
                                              2 * ghostCells)], 1e-300)
                                        << "Wrong input array at " << i << ", " << j;

                                auto updateCorrectValue = [&](int x, int y, int z) {

                                    double xOffsetValue = boundaryIndex(nx, i, x, 0, (nx - 1), 1);

                                    double yOffsetValue = boundaryIndex(ny, j, y, 0, (ny - 1) * nx, nx);


                                    double zOffsetValue = boundaryIndex(nz, k, z, 0, (nz - 1) * nx * ny, nx * ny);


                                    // The two tests below (in the if/else) is simply a test for the unit test...
                                    // setting this up correctly was really difficult.
                                    if (grid.getBoundaryCondition(0) == boundary::PERIODIC) {
                                        int iPlusH = i + x;
                                        int jPlusH = j + y;
                                        int kPlusH = k + z;

                                        while (iPlusH < 0) {
                                            iPlusH += nx;
                                        }


                                        while (jPlusH < 0) {
                                            jPlusH += ny;
                                        }

                                        if (kPlusH < 0) {
                                            kPlusH += nz;
                                        }


                                        iPlusH %= nx;
                                        jPlusH %= ny;
                                        kPlusH %= nz;

                                        int indexNewH = (iPlusH + ghostCells) + (jPlusH + ghostCells) *
                                            (nx + 2 * ghostCells) + (kPlusH + ghostCells) *
                                            (nx + 2 * ghostCells) * (ny + 2 * ghostCells);
                                        int indexNew = (i + ghostCells) + (j + ghostCells) * (nx + 2 * ghostCells) +
                                            (k + ghostCells) *
                                            (nx + 2 * ghostCells) * (ny + 2 * ghostCells);

                                        auto u = inputCopyOnCPU->getScalarMemoryArea(
                                                var)->getPointer()[indexNew];


                                        auto uH = inputCopyOnCPU->getScalarMemoryArea(
                                                var)->getPointer()[indexNewH];


                                        ASSERT_DOUBLE_EQ(u - uH, xOffsetValue + yOffsetValue + zOffsetValue);

                                    } else {
                                        int iPlusH = i + x;
                                        int jPlusH = j + y;
                                        int kPlusH = k + z;

                                        if (iPlusH < 0) {
                                            iPlusH = 0;
                                        } else if (iPlusH >= nx) {
                                            iPlusH = nx - 1;
                                        }


                                        if (jPlusH < 0) {
                                            jPlusH = 0;
                                        } else if (jPlusH >= nx) {
                                            jPlusH = nx - 1;
                                        }


                                        if (kPlusH < 0) {
                                            kPlusH = 0;
                                        } else if (kPlusH >= nx) {
                                            kPlusH = nx - 1;
                                        }

                                        int indexNewH = (iPlusH + ghostCells) + (jPlusH + ghostCells) *
                                            (nx + 2 * ghostCells) + (kPlusH + ghostCells) *
                                            (nx + 2 * ghostCells) * (ny + 2 * ghostCells);
                                        int indexNew = (i + ghostCells) + (j + ghostCells) * (nx + 2 * ghostCells) +
                                            (k + ghostCells) *
                                            (nx + 2 * ghostCells) * (ny + 2 * ghostCells);

                                        auto u = inputCopyOnCPU->getScalarMemoryArea(
                                                var)->getPointer()[indexNew];


                                        auto uH = inputCopyOnCPU->getScalarMemoryArea(
                                                var)->getPointer()[indexNewH];

                                        ASSERT_DOUBLE_EQ(u - uH, xOffsetValue + yOffsetValue + zOffsetValue);



                                    }

                                    correctValue += std::pow(std::abs(xOffsetValue + yOffsetValue + zOffsetValue),
                                            p);
                                };

                                for (auto x : {
                                        -h, h
                                        }) {

                                    for (int y = -h; y < h + 1; ++y) {
                                        for (int z = -h; z < h + 1; ++z) {
                                            // x side
                                            updateCorrectValue(x, y, z);


                                            if (y > -h && y < h)  {
                                                // y side
                                                updateCorrectValue(y, x, z);


                                                if (z > -h && z < h) {
                                                    // z side
                                                    updateCorrectValue(y, z, x);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    correctValue /= (nx * ny * nz);

                    const std::string boundaryName = grid.getBoundaryCondition(
                            0) == boundary::PERIODIC ? "periodic" : "neumann";

                    if ((platform == "cpu" && p < 2) || p < 2) {
                        ASSERT_DOUBLE_EQ(correctValue, copyOnCPU->getScalarMemoryArea(
                                var)->getPointer()[h]) <<
                                    "Wrong structure function with boundary condition " << boundaryName
                                    << ", variable " << copyOnCPU->getName(var) << ", h = " << h;
                    } else {
                        // Due to round off errors in summation (CUDA is actually MORE accurate here),
                        // we need to show some lenience
                        ASSERT_NEAR(correctValue, copyOnCPU->getScalarMemoryArea(
                                var)->getPointer()[h], 1e-8 * correctValue) <<
                                    "Wrong structure function with boundary condition " << boundaryName
                                    << ", variable " << copyOnCPU->getName(var) << ", h = " << h;
                    }

                }
            }
        }
    }

}

#ifdef ALSVINN_HAVE_CUDA
INSTANTIATE_TEST_CASE_P(TestConstant, StructureTestBoundary,
    ::testing::Values(
        StructureParameters("cpu", 1),
        StructureParameters("cpu", 2),
        StructureParameters("cpu", 3),
        StructureParameters("cpu", 4),
        StructureParameters("cpu", 5),
        StructureParameters("cpu", 6),
        StructureParameters("cpu", 7),
        StructureParameters("cpu", 8),
        StructureParameters("cuda", 1),
        StructureParameters("cuda", 2),
        StructureParameters("cuda", 3),
        StructureParameters("cuda", 4),
        StructureParameters("cuda", 5),
        StructureParameters("cuda", 6),
        StructureParameters("cuda", 7),
        StructureParameters("cuda", 8)
    ));
#else
INSTANTIATE_TEST_CASE_P(TestConstant, StructureTestBoundary,
    ::testing::Values(
        StructureParameters("cpu", 1),
        StructureParameters("cpu", 2),
        StructureParameters("cpu", 3),
        StructureParameters("cpu", 4),
        StructureParameters("cpu", 5),
        StructureParameters("cpu", 6),
        StructureParameters("cpu", 7),
        StructureParameters("cpu", 8)
    ));
#endif
