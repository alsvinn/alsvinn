#include <gtest/gtest.h>
#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/WENOCoefficients.hpp"
#include "alsfvm/equation/euler/EulerParameters.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

class CUDAEnoTest : public ::testing::Test {
public:
    size_t nx = 10;
    size_t ny = 10;
    size_t nz = 1;

    Grid grid;

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<MemoryFactory> memoryFactory;
    ReconstructionFactory reconstructionFactory;
    VolumeFactory volumeFactory;

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfigurationCPU;
    alsfvm::shared_ptr<MemoryFactory> memoryFactoryCPU;
    VolumeFactory volumeFactoryCPU;

    simulator::SimulatorParameters simulatorParameters;

    alsfvm::shared_ptr<Reconstruction> wenoCUDA;

    alsfvm::shared_ptr<Volume> conserved;
    alsfvm::shared_ptr<Volume> left;
    alsfvm::shared_ptr<Volume> right;

    alsfvm::shared_ptr<Volume> conservedCPU;
    alsfvm::shared_ptr<Volume> leftCPU;
    alsfvm::shared_ptr<Volume> rightCPU;



    CUDAEnoTest()
        : grid({ 0, 0, 0 }, { 1, 1, 1 }, ivec3(nx, ny, nz)),
        deviceConfiguration(new DeviceConfiguration("cuda")),
        memoryFactory(new MemoryFactory(deviceConfiguration)),
        volumeFactory("euler3", memoryFactory),
        deviceConfigurationCPU(new DeviceConfiguration("cpu")),
        memoryFactoryCPU(new MemoryFactory(deviceConfigurationCPU)),
        volumeFactoryCPU("euler3", memoryFactoryCPU)
    {
        auto eulerParameters = alsfvm::make_shared<equation::euler::EulerParameters>();

        simulatorParameters.setEquationParameters(eulerParameters);
    }

    void makeReconstruction(const std::string name, size_t newNx) {
        nx = newNx;
        nz = 1;
        ny = 1;

        grid = Grid({ 0, 0, 0 }, { 1, 1, 1 }, ivec3(nx, ny, nz));

        makeReconstruction(name);
    }

    void makeReconstruction(const std::string& name) {
        wenoCUDA = reconstructionFactory.createReconstruction(name, "euler3", simulatorParameters, memoryFactory, grid, deviceConfiguration);

        conserved = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        left = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        right = volumeFactory.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());

        conservedCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());
        leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz, wenoCUDA->getNumberOfGhostCells());

        conserved->makeZero();
    }

};

TEST_F(CUDAEnoTest, ConstantZeroTestSecondOrder) {
    makeReconstruction("eno2");

    wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);

    right->copyTo(*rightCPU);
    left->copyTo(*leftCPU);
    for_each_internal_volume_index(*leftCPU, 0, [&](size_t, size_t middle, size_t) {
        ASSERT_EQ(0, leftCPU->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, leftCPU->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, leftCPU->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, leftCPU->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, leftCPU->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(0, rightCPU->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, rightCPU->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, rightCPU->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, rightCPU->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, rightCPU->getScalarMemoryArea(4)->getPointer()[middle]);
    });
}
TEST_F(CUDAEnoTest, ConstantOneTestSecondOrder) {
    makeReconstruction("eno2");
    for_each_cell_index(*conservedCPU, [&](size_t index) {
        conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    conservedCPU->copyTo(*conserved);


    wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);

    left->copyTo(*leftCPU);
    right->copyTo(*rightCPU);

    for_each_internal_volume_index(*left, 0, [&](size_t, size_t middle, size_t) {
        ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, leftCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, leftCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

        ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1, rightCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, rightCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
    });
}


TEST_F(CUDAEnoTest, ReconstructionSimple) {
    makeReconstruction("eno2");

    for_each_cell_index(*conservedCPU, [&](size_t index) {
        // fill some dummy data
        conservedCPU->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conservedCPU->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });


    // This is the main ingredient:
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[1] = 2;
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[2] = 0;
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[3] = 1;

    conservedCPU->copyTo(*conserved);


    wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);
    cudaDeviceSynchronize();
    
    right->copyTo(*rightCPU);
    left->copyTo(*leftCPU);
    ASSERT_EQ(1.0 / 2.0, rightCPU->getScalarMemoryArea("rho")->getPointer()[2]);
    ASSERT_EQ(-1.0 / 2.0, leftCPU->getScalarMemoryArea("rho")->getPointer()[2]);
}
