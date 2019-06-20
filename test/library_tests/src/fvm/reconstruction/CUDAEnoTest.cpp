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

    int ngx = 0;
    int ngy = 0;
    int ngz = 0;


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
        : grid({ 0, 0, 0 }, {
        1, 1, 1
    }, ivec3(nx, ny, nz)),
    deviceConfiguration(new DeviceConfiguration("cuda")),
    memoryFactory(new MemoryFactory(deviceConfiguration)),
    volumeFactory("euler3", memoryFactory),
    deviceConfigurationCPU(new DeviceConfiguration("cpu")),
    memoryFactoryCPU(new MemoryFactory(deviceConfigurationCPU)),
    volumeFactoryCPU("euler3", memoryFactoryCPU) {
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
        wenoCUDA = reconstructionFactory.createReconstruction(name, "euler3",
                simulatorParameters, memoryFactory, grid, deviceConfiguration);

        ngx = wenoCUDA->getNumberOfGhostCells();
        ngy = wenoCUDA->getNumberOfGhostCells();
        conserved = volumeFactory.createConservedVolume(nx, ny, nz,
                wenoCUDA->getNumberOfGhostCells());
        left = volumeFactory.createConservedVolume(nx, ny, nz,
                wenoCUDA->getNumberOfGhostCells());
        right = volumeFactory.createConservedVolume(nx, ny, nz,
                wenoCUDA->getNumberOfGhostCells());

        conservedCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz,
                wenoCUDA->getNumberOfGhostCells());
        rightCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz,
                wenoCUDA->getNumberOfGhostCells());
        leftCPU = volumeFactoryCPU.createConservedVolume(nx, ny, nz,
                wenoCUDA->getNumberOfGhostCells());

        conserved->makeZero();
    }

};

TEST_F(CUDAEnoTest, ConstantZeroTestSecondOrder) {
    makeReconstruction("eno2");

    wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);

    right->copyTo(*rightCPU);
    left->copyTo(*leftCPU);

    for_each_cell_index(*leftCPU, [&](size_t middle) {
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
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
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

    for_each_cell_index(*left, [&](size_t middle) {
        ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1,  leftCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, leftCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);

        ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(1)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(2)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(1,  rightCPU->getScalarMemoryArea(3)->getPointer()[middle], 1e-8);
        ASSERT_NEAR(10, rightCPU->getScalarMemoryArea(4)->getPointer()[middle], 1e-8);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
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
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[4 * (nx + 4) + 1] = 2;
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[4 * (nx + 4) + 2] = 0;
    conservedCPU->getScalarMemoryArea("rho")->getPointer()[4 * (nx + 4) + 3] = 1;

    conservedCPU->copyTo(*conserved);


    wenoCUDA->performReconstruction(*conserved, 0, 0, *left, *right);
    cudaDeviceSynchronize();

    right->copyTo(*rightCPU);
    left->copyTo(*leftCPU);
    ASSERT_EQ(1.0 / 2.0, rightCPU->getScalarMemoryArea("rho")->getPointer()[4 *
              (nx + 4) + 2]);
    ASSERT_EQ(-1.0 / 2.0, leftCPU->getScalarMemoryArea("rho")->getPointer()[4 *
              (nx + 4) + 2]);
}
