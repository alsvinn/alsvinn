#include <gtest/gtest.h>

#include "alsfvm/reconstruction/ENOCPU.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/Euler.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;

TEST(EnoTest, ConstantZeroTestSecondOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
	ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());

    conserved->makeZero();

    

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_internal_volume_index(*left, 0, [&](size_t , size_t middle, size_t ) {
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
    });
}

TEST(EnoTest, ConstantOneTestSecondOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = std::make_shared<DeviceConfiguration>();
    auto memoryFactory = std::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("euler", memoryFactory);
	ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("rho")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mx")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("my")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("mz")->getPointer()[index] = 1;
        conserved->getScalarMemoryArea("E")->getPointer()[index] = 10;

    });
  

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_internal_volume_index(*left, 0, [&](size_t , size_t middle, size_t ) {
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
    });
}
