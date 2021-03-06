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

#include "alsfvm/reconstruction/ENOCPU.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/reconstruction/ENOCoefficients.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

TEST(EnoTest, CoefficientsTest) {
    // Check that all coefficients sum up to 1
    for (int r = 0; r < 3; r++) {
        ASSERT_EQ(1.0, ENOCoeffiecients<2>::coefficients[r][0]
            + ENOCoeffiecients<2>::coefficients[r][1])
                << " ENO Coefficients did not sum to 1 for r = " << r << ", order = 2";
    }

    for (int r = 0; r < 4; r++) {

        ASSERT_NEAR(1.0, ENOCoeffiecients<3>::coefficients[r][0]
            + ENOCoeffiecients<3>::coefficients[r][1]
            + ENOCoeffiecients<3>::coefficients[r][2],
            2e-7) << " ENO Coefficients did not sum to 1 for r = " << r << ", order = 3";
    }
}

TEST(EnoTest, ConstantZeroTestSecondOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());

    conserved->makeZero();


    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;
    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_cell_index(*left, [&](size_t middle ) {
        for (int var = 0; var < 5; ++var) {
            ASSERT_EQ(0, left->getScalarMemoryArea(var)->getPointer()[middle])
                    << "Failed at index = " << middle << " var = " << var;

            ASSERT_EQ(0, right->getScalarMemoryArea(var)->getPointer()[middle])
                    << "Failed at index = " << middle << " var = " << var;
        }
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}


TEST(EnoTest, ConstantZeroTestThirdOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<3> enoCPU(memoryFactory, nx, ny, nz);

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());

    conserved->makeZero();



    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);
    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*left, [&]( size_t middle ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, left->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(0, right->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(0, right->getScalarMemoryArea(4)->getPointer()[middle]);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(EnoTest, ConstantOneTestSecondOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });


    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);


    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_EQ(1, left->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(1, left->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(1, left->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(1, left->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(10, left->getScalarMemoryArea(4)->getPointer()[middle]);

        ASSERT_EQ(1, right->getScalarMemoryArea(0)->getPointer()[middle]);
        ASSERT_EQ(1, right->getScalarMemoryArea(1)->getPointer()[middle]);
        ASSERT_EQ(1, right->getScalarMemoryArea(2)->getPointer()[middle]);
        ASSERT_EQ(1, right->getScalarMemoryArea(3)->getPointer()[middle]);
        ASSERT_EQ(10, right->getScalarMemoryArea(4)->getPointer()[middle]);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(EnoTest, ConstantOneTestThirdOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<3> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });


    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);



    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;

    for_each_cell_index(*left, [&]( size_t middle ) {
        ASSERT_NEAR(1, left->getScalarMemoryArea(0)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(1, left->getScalarMemoryArea(1)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(1, left->getScalarMemoryArea(2)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(1, left->getScalarMemoryArea(3)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(10, left->getScalarMemoryArea(4)->getPointer()[middle], 2e-6);

        ASSERT_NEAR(1, right->getScalarMemoryArea(0)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(1, right->getScalarMemoryArea(1)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(1, right->getScalarMemoryArea(2)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(1, right->getScalarMemoryArea(3)->getPointer()[middle], 2e-6);
        ASSERT_NEAR(10, right->getScalarMemoryArea(4)->getPointer()[middle], 2e-6);
    }, {ngx - 1, ngy, ngz}, {ngx - 1, ngy, ngz});
}

TEST(EnoTest, ReconstructionSimple) {
    const size_t nx = 10, ny = 1, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[1] = 2;
    conserved->getScalarMemoryArea("rho")->getPointer()[2] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3] = 1;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    ASSERT_EQ(1.0 / 2.0, right->getScalarMemoryArea("rho")->getPointer()[2]);
    ASSERT_EQ(-1.0 / 2.0, left->getScalarMemoryArea("rho")->getPointer()[2]);

}

TEST(EnoTest, ReconstructionSimpleReverse) {
    const size_t nx = 10, ny = 1, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[1] = 1;
    conserved->getScalarMemoryArea("rho")->getPointer()[2] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3] = 2;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    ASSERT_EQ(-1.0 / 2.0, right->getScalarMemoryArea("rho")->getPointer()[2]);
    ASSERT_EQ(1.0 / 2.0, left->getScalarMemoryArea("rho")->getPointer()[2]);

}

TEST(EnoTest, ReconstructionSimpleReverseYDirection) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler3", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz,
            enoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });

    const size_t totalNx = nx + 2 * enoCPU.getNumberOfGhostCells();
    // This is the main ingredient:
    conserved->getScalarMemoryArea("rho")->getPointer()[3 + 1 * totalNx] = 1;
    conserved->getScalarMemoryArea("rho")->getPointer()[3 + 2 * totalNx] = 0;
    conserved->getScalarMemoryArea("rho")->getPointer()[3 + 3 * totalNx] = 2;
    enoCPU.performReconstruction(*conserved, 1, 0, *left, *right);

    ASSERT_EQ(-1.0 / 2.0, right->getScalarMemoryArea("rho")->getPointer()[3 + 2 *
              totalNx]);
    ASSERT_EQ(1.0 / 2.0, left->getScalarMemoryArea("rho")->getPointer()[3 + 2 *
              totalNx]);

}
