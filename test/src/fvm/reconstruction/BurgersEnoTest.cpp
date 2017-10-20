#include <gtest/gtest.h>

#include "alsfvm/reconstruction/ENOCPU.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/reconstruction/ENOCoefficients.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::reconstruction;
using namespace alsfvm::grid;



TEST(BurgersEnoTest, ConstantZeroTestSecondOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
	ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());

    conserved->makeZero();

    
    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle])
                << "Failed at index " << middle;
    }, {ngx-1, ngy, ngz}, {-ngx+1, -ngy, -ngz});
}


TEST(BurgersEnoTest, ConstantZeroTestThirdOrder) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
    ENOCPU<3> enoCPU(memoryFactory, nx, ny, nz);

    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());

    conserved->makeZero();


    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_EQ(0, left->getScalarMemoryArea(0)->getPointer()[middle]);

        ASSERT_EQ(0, right->getScalarMemoryArea(0)->getPointer()[middle]);
    }, {ngx-1, ngy, ngz}, {-ngx+1, -ngy, -ngz});
}

TEST(BurgersEnoTest, ConstantOneTestSecondOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
	ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
	auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
	auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());

    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("u")->getPointer()[index] = 1;
    });
  

    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_cell_index(*left, [&](size_t middle ) {
        ASSERT_EQ(1, left->getScalarMemoryArea(0)->getPointer()[middle]);

        ASSERT_EQ(1, right->getScalarMemoryArea(0)->getPointer()[middle]);
    }, {ngx-1, ngy, ngz}, {-ngx+1, -ngy, -ngz});
}

TEST(BurgersEnoTest, ConstantOneTestThirdOrder) {

    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
    ENOCPU<3> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    const int ngx = enoCPU.getNumberOfGhostCells();
    const int ngy = enoCPU.getNumberOfGhostCells();
    const int ngz = 0;
    for_each_cell_index(*conserved, [&] (size_t index) {
        conserved->getScalarMemoryArea("u")->getPointer()[index] = 1;
    });


    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    for_each_cell_index(*left, [&]( size_t middle ) {
        ASSERT_NEAR(1, left->getScalarMemoryArea(0)->getPointer()[middle], 1e-8);
    }, {ngx-1, ngy, ngz}, {-ngx+1, -ngy, -ngz});
}

TEST(BurgersEnoTest, ReconstructionSimple) {
    const size_t nx = 10, ny = 1, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("u")->getPointer()[index] = 1;
    });

    // This is the main ingredient:
    conserved->getScalarMemoryArea("u")->getPointer()[1] = 2;
    conserved->getScalarMemoryArea("u")->getPointer()[2] = 0;
    conserved->getScalarMemoryArea("u")->getPointer()[3] = 1;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    ASSERT_EQ(1.0/2.0, right->getScalarMemoryArea("u")->getPointer()[2]);
    ASSERT_EQ(-1.0/2.0, left->getScalarMemoryArea("u")->getPointer()[2]);

}

TEST(BurgersEnoTest, ReconstructionSimpleReverse) {
    const size_t nx = 10, ny = 1, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("u")->getPointer()[index] = 1;
    });

    // This is the main ingredient:
    conserved->getScalarMemoryArea("u")->getPointer()[1] = 1;
    conserved->getScalarMemoryArea("u")->getPointer()[2] = 0;
    conserved->getScalarMemoryArea("u")->getPointer()[3] = 2;

    enoCPU.performReconstruction(*conserved, 0, 0, *left, *right);

    ASSERT_EQ(-1.0/2.0, right->getScalarMemoryArea("u")->getPointer()[2]);
    ASSERT_EQ(1.0/2.0, left->getScalarMemoryArea("u")->getPointer()[2]);

}

TEST(BurgersEnoTest, ReconstructionSimpleReverseYDirection) {
    const size_t nx = 10, ny = 10, nz = 1;

    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>();
    auto memoryFactory = alsfvm::make_shared<MemoryFactory>(deviceConfiguration);

    VolumeFactory volumeFactory("burgers", memoryFactory);
    ENOCPU<2> enoCPU(memoryFactory, nx, ny, nz);
    auto conserved = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto left = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    auto right = volumeFactory.createConservedVolume(nx, ny, nz, enoCPU.getNumberOfGhostCells());
    for_each_cell_index(*conserved, [&] (size_t index) {
        // fill some dummy data
        conserved->getScalarMemoryArea("u")->getPointer()[index] = 1;
    });

    const size_t totalNx = nx + 2 * enoCPU.getNumberOfGhostCells();
    // This is the main ingredient:
    conserved->getScalarMemoryArea("u")->getPointer()[3+1 * totalNx] = 1;
    conserved->getScalarMemoryArea("u")->getPointer()[3+2 * totalNx] = 0;
    conserved->getScalarMemoryArea("u")->getPointer()[3+3 * totalNx] = 2;
    enoCPU.performReconstruction(*conserved, 1, 0, *left, *right);

    ASSERT_EQ(-1.0/2.0, right->getScalarMemoryArea("u")->getPointer()[3+2 * totalNx]);
    ASSERT_EQ(1.0/2.0, left->getScalarMemoryArea("u")->getPointer()[3 + 2 * totalNx]);

}
