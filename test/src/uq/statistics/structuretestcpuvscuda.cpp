#include <gtest/gtest.h>
#include "alsutils/config.hpp"
#ifdef ALSVINN_HAVE_CUDA

#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsuq/stats/StatisticsFactory.hpp"
#include "alsfvm/volume/make_volume.hpp"

#include <random>

class TestWriter : public alsfvm::io::Writer {
public:

    TestWriter(alsfvm::volume::Volume& conservedVariablesSaved,
        alsfvm::volume::Volume& extraVariablesSaved) :
        conservedVariablesSaved(conservedVariablesSaved),
        extraVariablesSaved(extraVariablesSaved) {

    }


    static std::shared_ptr<alsfvm::io::Writer> makeInstance(
        alsfvm::volume::Volume& conservedVariablesSaved,
        alsfvm::volume::Volume& extraVariablesSaved) {
        std::shared_ptr<alsfvm::io::Writer> pointer;
        pointer.reset(new TestWriter(conservedVariablesSaved, extraVariablesSaved));
        return pointer;
    }



    alsfvm::volume::Volume& conservedVariablesSaved;
    alsfvm::volume::Volume& extraVariablesSaved;

    void write(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables, const alsfvm::grid::Grid&,
        const alsfvm::simulator::TimestepInformation&) override  {

        conservedVariables.copyTo(conservedVariablesSaved);
        extraVariables.copyTo(extraVariablesSaved);
    }
};

class StructureTestCPUvsCUDA : public ::testing::Test {
public:

    const int ghostCells = 3;
    const alsfvm::ivec3 innerSize = {16, 16, 16};
    const std::string equation = "euler3";

    const int p = 2;
    const int numberOfH = 8;

    std::shared_ptr<alsuq::mpi::Configuration> mpiConfigurationCPU =
        std::make_shared<alsuq::mpi::Configuration>
        (MPI_COMM_WORLD, "cpu");


    std::shared_ptr<alsuq::mpi::Configuration> mpiConfigurationCUDA =
        std::make_shared<alsuq::mpi::Configuration>
        (MPI_COMM_WORLD, "cuda");

    std::shared_ptr<alsfvm::DeviceConfiguration> deviceConfigurationCPU =
        std::make_shared<alsfvm::DeviceConfiguration>("cpu");
    std::shared_ptr<alsfvm::DeviceConfiguration> deviceConfigurationCUDA =
        std::make_shared<alsfvm::DeviceConfiguration>("cuda");

    std::shared_ptr<alsfvm::memory::MemoryFactory> memoryFactoryCPU =
        std::make_shared<alsfvm::memory::MemoryFactory>(deviceConfigurationCPU);
    std::shared_ptr<alsfvm::memory::MemoryFactory> memoryFactoryCUDA =
        std::make_shared<alsfvm::memory::MemoryFactory>(deviceConfigurationCUDA);


    std::shared_ptr<alsfvm::volume::Volume> volumeConservedCPU =
        alsfvm::volume::makeConservedVolume("cpu", equation, innerSize, ghostCells);
    std::shared_ptr<alsfvm::volume::Volume> volumeExtraCPU =
        alsfvm::volume::makeExtraVolume("cpu", equation, innerSize, ghostCells);

    std::shared_ptr<alsfvm::volume::Volume> volumeConservedCUDA =
        alsfvm::volume::makeConservedVolume("cuda", equation, innerSize, ghostCells);
    std::shared_ptr<alsfvm::volume::Volume> volumeExtraCUDA =
        alsfvm::volume::makeExtraVolume("cuda", equation, innerSize, ghostCells);

    std::shared_ptr<alsfvm::volume::Volume> conservedOutputCPU =
        alsfvm::volume::makeConservedVolume("cpu", equation, {numberOfH, 1, 1},
            0);
    std::shared_ptr<alsfvm::volume::Volume> extraOutputCPU =
        alsfvm::volume::makeConservedVolume("cpu", equation, {numberOfH, 1, 1},
            0);

    // yes, these two should be on the CPU
    std::shared_ptr<alsfvm::volume::Volume> conservedOutputCUDA =
        alsfvm::volume::makeConservedVolume("cpu", equation, {numberOfH, 1, 1},
            0);
    std::shared_ptr<alsfvm::volume::Volume> extraOutputCUDA =
        alsfvm::volume::makeConservedVolume("cpu", equation, {numberOfH, 1, 1},
            0);

    std::shared_ptr<alsfvm::io::Writer> cpuWriter = TestWriter::makeInstance(
            *conservedOutputCPU, *extraOutputCPU);
    std::shared_ptr<alsfvm::io::Writer> cudaWriter = TestWriter::makeInstance(
            *conservedOutputCUDA, *extraOutputCUDA);

    std::vector<std::shared_ptr<alsfvm::volume::Volume>> volumes = {
        volumeConservedCPU,
        volumeExtraCPU,
        volumeConservedCUDA,
        volumeExtraCUDA
    };

    std::vector<std::shared_ptr<alsfvm::volume::Volume>> volumesCPU = {
        volumeConservedCPU,
        volumeExtraCPU,
    };

    std::vector<std::shared_ptr<alsfvm::volume::Volume>> volumesCUDA = {
        volumeConservedCUDA,
        volumeExtraCUDA
    };

    std::vector<std::shared_ptr<alsfvm::volume::Volume>> volumesOutputCPU = {
        conservedOutputCPU,
        extraOutputCPU,
    };

    std::vector<std::shared_ptr<alsfvm::volume::Volume>> volumesOutputCUDA = {
        conservedOutputCUDA,
        extraOutputCUDA
    };

    alsuq::stats::StatisticsFactory statisticsFactory;

    std::shared_ptr<alsuq::stats::Statistics> structureCPU;
    std::shared_ptr<alsuq::stats::Statistics> structureCUDA;

    alsfvm::grid::Grid grid{alsfvm::rvec3{0, 0, 0},
               alsfvm::rvec3{1, 1, 1},
               innerSize};

    alsfvm::simulator::TimestepInformation timestepInformation;

    StructureTestCPUvsCUDA() {
        for (auto& volumePtr : volumes) {
            volumePtr->makeZero();
        }

        boost::property_tree::ptree ptreeParameters;
        ptreeParameters.put("p", p);
        ptreeParameters.put("numberOfH", numberOfH);


        alsuq::stats::StatisticsParameters parametersCPU(ptreeParameters);
        parametersCPU.setMpiConfiguration(mpiConfigurationCPU);
        parametersCPU.setNumberOfSamples(1);


        structureCPU = statisticsFactory.makeStatistics("cpu",
                "structure_cube",
                parametersCPU);

        for (auto statsName :
            structureCPU->getStatisticsNames()) {

            structureCPU->addWriter(statsName, cpuWriter);
        }

        alsuq::stats::StatisticsParameters parametersCUDA(ptreeParameters);
        parametersCUDA.setMpiConfiguration(mpiConfigurationCPU);
        parametersCUDA.setNumberOfSamples(1);

        structureCUDA = statisticsFactory.makeStatistics("cuda",
                "structure_cube",
                parametersCUDA);

        for (auto statsName :
            structureCUDA->getStatisticsNames()) {

            structureCUDA->addWriter(statsName, cudaWriter);
        }
    }

    void checkEqual(const alsfvm::volume::Volume& a,
        const alsfvm::volume::Volume& b) {

        ASSERT_EQ(a.getTotalNumberOfXCells(), b.getTotalNumberOfXCells());
        ASSERT_EQ(a.getTotalNumberOfYCells(), b.getTotalNumberOfYCells());
        ASSERT_EQ(a.getTotalNumberOfZCells(), b.getTotalNumberOfZCells());

        ASSERT_EQ(a.getTotalNumberOfXCells(), numberOfH);
        ASSERT_EQ(a.getTotalNumberOfYCells(), 1);
        ASSERT_EQ(a.getTotalNumberOfZCells(), 1);

        for (size_t var = 0; var < a.getNumberOfVariables(); ++var) {
            for (int h = 0; h < numberOfH; ++h) {
                ASSERT_EQ(a.getScalarMemoryArea(var)->getPointer()[h],
                    b.getScalarMemoryArea(var)->getPointer()[h])
                        << "Not equal at variable = " << a.getName(var) << ", h = " << h << "\n";

            }
        }
    }

    void checkCPUVolumeEqualToCUDAVolume() {
        for (size_t i = 0; i < volumesOutputCPU.size(); ++i) {
            checkEqual(*volumesOutputCPU[i], *volumesOutputCUDA[i]);
        }
    }
};

TEST_F(StructureTestCPUvsCUDA, TestConstant) {

    structureCPU->write(*volumeConservedCPU, *volumeExtraCPU, grid,
        timestepInformation);

    structureCPU->combineStatistics();
    structureCPU->finalizeStatistics();
    structureCPU->writeStatistics(grid);

    structureCUDA->write(*volumeConservedCUDA, *volumeExtraCUDA, grid,
        timestepInformation);


    structureCUDA->combineStatistics();
    structureCUDA->finalizeStatistics();
    structureCUDA->writeStatistics(grid);


    checkCPUVolumeEqualToCUDAVolume();

}

TEST_F(StructureTestCPUvsCUDA, TestIndex) {
    // Cell i,j,k gets the value k * nx * ny + j * nx + i

    std::vector<double> input(volumeConservedCPU->getTotalNumberOfXCells()
        *volumeConservedCPU->getTotalNumberOfYCells()
        *volumeConservedCPU->getTotalNumberOfZCells(), 0.0);


    for (size_t k = ghostCells; k < innerSize.z + ghostCells; ++k) {
        for (size_t j = ghostCells; j < innerSize.y + ghostCells; ++j) {
            for (size_t i = ghostCells; i < innerSize.z + ghostCells; ++i) {
                size_t index = k * conservedOutputCPU->getTotalNumberOfXCells() *
                    conservedOutputCPU->getTotalNumberOfYCells() +
                    j * conservedOutputCPU->getTotalNumberOfXCells() + i;

                input[index] = index;
            }
        }
    }


    for (size_t volumeIndex = 0; volumeIndex < volumes.size(); ++volumeIndex) {
        for (size_t var = 0; var < volumes[volumeIndex]->getNumberOfVariables();
            ++var) {
            volumes[volumeIndex]->getScalarMemoryArea(var)->copyFromHost(input.data(),
                input.size());
        }
    }


    structureCPU->write(*volumeConservedCPU, *volumeExtraCPU, grid,
        timestepInformation);

    structureCPU->combineStatistics();
    structureCPU->finalizeStatistics();
    structureCPU->writeStatistics(grid);

    structureCUDA->write(*volumeConservedCUDA, *volumeExtraCUDA, grid,
        timestepInformation);


    structureCUDA->combineStatistics();
    structureCUDA->finalizeStatistics();
    structureCUDA->writeStatistics(grid);


    checkCPUVolumeEqualToCUDAVolume();



}



TEST_F(StructureTestCPUvsCUDA, TestRandom) {
    // Generates ten random samples and checks that the output becomes the same
    // on both cpu and gpu
    const int numberOfSamples = 10;
    std::random_device randomDevice;
    std::mt19937 genenerator(
        randomDevice()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> distribution(1.0, 2.0);

    for (int sample = 0; sample < numberOfSamples; ++sample) {
        for (size_t volumeIndex = 0; volumeIndex < volumes.size(); ++volumeIndex) {
            for (size_t var = 0; var < volumes[volumeIndex]->getNumberOfVariables();
                ++var) {

                std::vector<double> input(volumeConservedCPU->getTotalNumberOfXCells()
                    *volumeConservedCPU->getTotalNumberOfYCells()
                    *volumeConservedCPU->getTotalNumberOfZCells(), 0.0);


                for (size_t k = ghostCells; k < innerSize.z + ghostCells; ++k) {
                    for (size_t j = ghostCells; j < innerSize.y + ghostCells; ++j) {
                        for (size_t i = ghostCells; i < innerSize.z + ghostCells; ++i) {
                            size_t index = k * conservedOutputCPU->getTotalNumberOfXCells() *
                                conservedOutputCPU->getTotalNumberOfYCells() +
                                j * conservedOutputCPU->getTotalNumberOfXCells() + i;

                            input[index] = distribution(genenerator);
                        }
                    }
                }



                volumes[volumeIndex]->getScalarMemoryArea(var)->copyFromHost(input.data(),
                    input.size());
            }
        }


        structureCPU->write(*volumeConservedCPU, *volumeExtraCPU, grid,
            timestepInformation);

        structureCUDA->write(*volumeConservedCUDA, *volumeExtraCUDA, grid,
            timestepInformation);

    }


    structureCPU->combineStatistics();
    structureCPU->finalizeStatistics();
    structureCPU->writeStatistics(grid);


    structureCUDA->combineStatistics();
    structureCUDA->finalizeStatistics();
    structureCUDA->writeStatistics(grid);


    checkCPUVolumeEqualToCUDAVolume();
}


#endif
