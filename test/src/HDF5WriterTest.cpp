#include "gtest/gtest.h"
#include <vector>
#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/io/hdf5_utils.hpp"
#include "alsfvm/io/io_utils.hpp"

using namespace alsfvm::io;
using namespace alsfvm::memory;
using namespace alsfvm::volume;
using namespace alsfvm;

class HDF5WriterTest : public ::testing::Test {
public:
    size_t nx, ny, nz;
    std::string basename;

    alsfvm::simulator::TimestepInformation info;

    std::shared_ptr<alsfvm::DeviceConfiguration> deviceConfiguration;
    std::shared_ptr<MemoryFactory> memoryFactory;
    std::vector<std::string> namesConserved;
    std::vector<std::string> namesExtra;
    Volume conservedVariables;
    Volume extraVariables;
    HDF5Writer writer;
	alsfvm::grid::Grid grid;
    HDF5WriterTest()
        : nx(10), ny(10), nz(10), basename("base"),
          deviceConfiguration(new alsfvm::DeviceConfiguration("cpu")),
          memoryFactory(new MemoryFactory(deviceConfiguration)),
          namesConserved({"alpha", "beta", "gamma"}),
          namesExtra({"rho", "phi"}),
          conservedVariables(namesConserved, memoryFactory, nx, ny, nz),
          extraVariables(namesExtra, memoryFactory, nx, ny, nz),
          writer(basename), grid(rvec3(0, 0, 0), rvec3(12.5, 13.5, 10.25), ivec3(nx, nx, nx))
    {

    }
};

TEST_F(HDF5WriterTest, ConstructTest) {
    writer.write(conservedVariables, extraVariables, grid,  info);
}

TEST_F(HDF5WriterTest, WriteAndReadTest) {

    // Write some dummy data
    for(size_t i = 0; i < namesConserved.size(); i++) {
        auto memoryArea = conservedVariables.getScalarMemoryArea(i);
        for(size_t j = 0; j < memoryArea->getSize(); j++) {
            memoryArea->getPointer()[j] = 1<<i + j;
        }
    }

    // Write some dummy data
    for(size_t i = 0; i < namesExtra.size(); i++) {
        auto memoryArea = extraVariables.getScalarMemoryArea(i);
        for(size_t j = 0; j < memoryArea->getSize(); j++) {
            memoryArea->getPointer()[j] = 2<<i + j;
        }
    }

    writer.write(conservedVariables, extraVariables, grid, info);

    // Now we will read back
    const std::string outputFilename = alsfvm::io::getOutputname(basename, 0)
            + std::string(".h5");

    HDF5Resource file( H5Fopen(outputFilename.c_str(),  H5F_ACC_RDWR, H5P_DEFAULT),
                           H5Fclose);


    for(size_t i = 0; i < namesConserved.size(); i++) {
        ASSERT_GT(H5Oexists_by_name(file.hid(), namesConserved[i].c_str(), H5P_DEFAULT), -1);
    }

    for(size_t i = 0; i < namesExtra.size(); i++) {
        ASSERT_GT(H5Oexists_by_name(file.hid(), namesExtra[i].c_str(), H5P_DEFAULT), -1);
    }


    for(size_t i = 0; i < namesConserved.size(); i++) {
        std::vector<double> data(nx*ny*nz);
        HDF5Resource dataset(H5Dopen2(file.hid(), namesConserved[i].c_str(), H5P_DEFAULT), H5Dclose);
        HDF5_SAFE_CALL(H5Dread(dataset.hid(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           data.data()));

        for(size_t j = 0; j < data.size(); j++) {
            ASSERT_EQ(1<<i + j, data[j]);
        }
    }

    for(size_t i = 0; i < namesExtra.size(); i++) {
        std::vector<double> data(nx*ny*nz);
        HDF5Resource dataset(H5Dopen2(file.hid(), namesExtra[i].c_str(), H5P_DEFAULT), H5Dclose);
        HDF5_SAFE_CALL(H5Dread(dataset.hid(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                           data.data()));

        for(size_t j = 0; j < data.size(); j++) {
            ASSERT_EQ(2<<i + j, data[j]);
        }
    }

    // We need to check the grid
    HDF5Resource gridGroup(H5Gopen2(file.hid(), "grid", H5P_DEFAULT), H5Gclose);

    HDF5Resource lowerCornerAttribute(H5Aopen(gridGroup.hid(), "vsLowerBounds", H5P_DEFAULT), H5Aclose);
    std::vector<float> lowerCorner(3,42);
    HDF5_SAFE_CALL(H5Aread(lowerCornerAttribute.hid(), H5T_NATIVE_FLOAT, lowerCorner.data()));

    ASSERT_EQ(lowerCorner[0], grid.getOrigin().x);
    ASSERT_EQ(lowerCorner[1], grid.getOrigin().y);
    ASSERT_EQ(lowerCorner[2], grid.getOrigin().z);

    HDF5Resource upperCornerAttribute(H5Aopen(gridGroup.hid(), "vsUpperBounds", H5P_DEFAULT), H5Aclose);
    std::vector<float> upperCorner(3,42);
    HDF5_SAFE_CALL(H5Aread(upperCornerAttribute.hid(), H5T_NATIVE_FLOAT, upperCorner.data()));

    ASSERT_EQ(upperCorner[0], grid.getTop().x);
    ASSERT_EQ(upperCorner[1], grid.getTop().y);
    ASSERT_EQ(upperCorner[2], grid.getTop().z);

    HDF5Resource startCellAttribute(H5Aopen(gridGroup.hid(), "vsStartCell", H5P_DEFAULT), H5Aclose);
    std::vector<int> startCell(3,42);
    HDF5_SAFE_CALL(H5Aread(startCellAttribute.hid(), H5T_NATIVE_INT, startCell.data()));

    ASSERT_EQ(startCell[0], 0);
    ASSERT_EQ(startCell[1], 0);
    ASSERT_EQ(startCell[2], 0);

    HDF5Resource numCellsAttribute(H5Aopen(gridGroup.hid(), "vsNumCells", H5P_DEFAULT), H5Aclose);
    std::vector<int> numCells(3,42);
    HDF5_SAFE_CALL(H5Aread(numCellsAttribute.hid(), H5T_NATIVE_INT, numCells.data()));

    ASSERT_EQ(numCells[0], grid.getDimensions().x);
    ASSERT_EQ(numCells[1], grid.getDimensions().y);
    ASSERT_EQ(numCells[2], grid.getDimensions().z);
}

