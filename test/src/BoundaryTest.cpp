#include <gtest/gtest.h>
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/boundary/BoundaryFactory.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/euler/AllVariables.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm::boundary;

struct BoundaryTest : public ::testing::Test {
    boost::shared_ptr<DeviceConfiguration> deviceConfiguration;
    boost::shared_ptr<MemoryFactory> memoryFactory;
    std::string equation = "euler";
    boost::shared_ptr<VolumeFactory> volumeFactory;
    size_t nx = 10, ny=11, nz=12;
    size_t ghostCells = 2;
    rvec3 lowerCorner = rvec3(0,0,0);
    rvec3 upperCorner = rvec3(1,1,1);
    ivec3 dimensions = ivec3(nx, ny, nz);

    grid::Grid grid;
    boost::shared_ptr<BoundaryFactory> boundaryFactory;
    BoundaryTest()
        : deviceConfiguration(new DeviceConfiguration("cpu")),
          memoryFactory(new MemoryFactory(deviceConfiguration)),
          volumeFactory(new VolumeFactory(equation, memoryFactory)),
          grid(lowerCorner, upperCorner, dimensions)
    {

    }


};


TEST_F(BoundaryTest, NeumannTest2CellsConstant) {
    boundaryFactory.reset(new BoundaryFactory("neumann", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, nz, ghostCells);
    volume->makeZero();
    const real C = 10;
    fill_volume<equation::euler::ConservedVariables>(*volume, grid, [&](real x, real y, real z, equation::euler::ConservedVariables& out) {
        out.E = C;
        out.m.x = C;
        out.m.y = C;
        out.m.z = C;
        out.rho = C;
    });

    auto boundary = boundaryFactory->createBoundary(ghostCells);

    boundary->applyBoundaryConditions(*volume, grid);


    for_each_cell_index(*volume, [&](size_t index) {
        ASSERT_EQ(volume->getScalarMemoryArea("rho")->getPointer()[index], C);
    });
}


TEST_F(BoundaryTest, NeumannTest2CellsVarying) {
    boundaryFactory.reset(new BoundaryFactory("neumann", deviceConfiguration));

    auto volume = volumeFactory->createConservedVolume(nx, ny, nz, ghostCells);
    volume->makeZero();

    auto rho = volume->getScalarMemoryArea("rho")->getView();
    auto mx = volume->getScalarMemoryArea("mx")->getView();
    auto my = volume->getScalarMemoryArea("my")->getView();
    auto mz = volume->getScalarMemoryArea("mz")->getView();
    auto E = volume->getScalarMemoryArea("E")->getView();

    for(size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for(size_t y = ghostCells; y < ny + ghostCells; ++y) {
            for(size_t z = ghostCells; z < nz + ghostCells; ++z) {
                rho.at(x, y, z) = rho.index(x, y, z);
                mx.at(x, y, z) = rho.index(x, y, z);
                my.at(x, y, z) = rho.index(x, y, z);
                mz.at(x, y, z) = rho.index(x, y, z);
                E.at(x, y, z) = rho.index(x, y, z);
            }
        }
    }


    auto boundary = boundaryFactory->createBoundary(ghostCells);

    boundary->applyBoundaryConditions(*volume, grid);

    // X side
    for(size_t y = ghostCells; y < ny + ghostCells; ++y) {
        for(size_t z = ghostCells; z < nz + ghostCells; ++z) {
            ASSERT_EQ(rho.at(0, y, z), rho.index(3, y, z));
            ASSERT_EQ(rho.at(1, y, z), rho.index(2, y, z));

            ASSERT_EQ(rho.at(ghostCells + nx + 1, y, z), rho.index(nx + ghostCells - 2, y, z));
            ASSERT_EQ(rho.at(ghostCells + nx, y, z), rho.index(nx + ghostCells - 1, y, z));
        }
    }

    // Y side
    for(size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for(size_t z = ghostCells; z < nz + ghostCells; ++z) {
            ASSERT_EQ(rho.at(x, 0, z), rho.index(x, 3, z));
            ASSERT_EQ(rho.at(x, 1, z), rho.index(x, 2, z));

            ASSERT_EQ(rho.at(x, ghostCells + ny + 1, z), rho.index(x, ny + ghostCells - 2, z));
            ASSERT_EQ(rho.at(x, ghostCells + ny, z), rho.index(x, ny + ghostCells - 1, z));
        }
    }

    // Z side
    for(size_t x = ghostCells; x < nx + ghostCells; ++x) {
        for(size_t y = ghostCells; y < ny + ghostCells; ++y) {
            ASSERT_EQ(rho.at(x, y, 0), rho.index(x, y, 3));
            ASSERT_EQ(rho.at(x, y, 1), rho.index(x, y, 2));

            ASSERT_EQ(rho.at(x, y, ghostCells + nz + 1), rho.index(x, y, nz + ghostCells - 2));
            ASSERT_EQ(rho.at(x, y, ghostCells + nz), rho.index(x, y, nz + ghostCells - 1));
        }
    }
}
