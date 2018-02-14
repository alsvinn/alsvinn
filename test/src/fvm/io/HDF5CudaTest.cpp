#include "gtest/gtest.h"
#ifdef ALSVINN_HAVE_CUDA
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

class HDF5CudaTest : public ::testing::Test {
public:
    size_t nx, ny, nz;
    std::string basename;

    alsfvm::simulator::TimestepInformation info;

    alsfvm::shared_ptr<alsfvm::DeviceConfiguration> deviceConfiguration;
    alsfvm::shared_ptr<MemoryFactory> memoryFactory;
    std::vector<std::string> namesConserved;
    std::vector<std::string> namesExtra;
    Volume conservedVariables;
    Volume extraVariables;
    HDF5Writer writer;
    alsfvm::grid::Grid grid;
    HDF5CudaTest()
        : nx(10), ny(10), nz(10), basename("base_cuda"),
          deviceConfiguration(new alsfvm::DeviceConfiguration("cuda")),
          memoryFactory(new MemoryFactory(deviceConfiguration)),
          namesConserved({ "alpha", "beta", "gamma" }),
    namesExtra({ "rho", "phi" }),
    conservedVariables(namesConserved, memoryFactory, nx, ny, nz),
    extraVariables(namesExtra, memoryFactory, nx, ny, nz),
    writer(basename), grid(rvec3(0, 0, 0), rvec3(10, 10, 10), ivec3(nx, nx, nx)) {

    }
};

TEST_F(HDF5CudaTest, ConstructTest) {
    writer.write(conservedVariables, extraVariables, grid, info);
}

TEST_F(HDF5CudaTest, WriteAndReadTest) {

    // Write some dummy data
    for (size_t i = 0; i < namesConserved.size(); i++) {
        auto memoryArea = conservedVariables.getScalarMemoryArea(i);
        std::vector<alsfvm::real> hostMemory(memoryArea->getSize());

        for (size_t j = 0; j < memoryArea->getSize(); j++) {
            hostMemory[j] = (1 << i) + j;
        }

        memoryArea->copyFromHost(hostMemory.data(), hostMemory.size());

    }

    // Write some dummy data
    for (size_t i = 0; i < namesExtra.size(); i++) {
        auto memoryArea = extraVariables.getScalarMemoryArea(i);
        std::vector<alsfvm::real> hostMemory(memoryArea->getSize());

        for (size_t j = 0; j < memoryArea->getSize(); j++) {
            hostMemory[j] = (2 << i) + j;
        }

        memoryArea->copyFromHost(hostMemory.data(), hostMemory.size());
    }

    writer.write(conservedVariables, extraVariables, grid, info);

    // Now we will read back
    const std::string outputFilename = alsfvm::io::getOutputname(basename, 0)
        + std::string(".h5");

    HDF5Resource file(H5Fopen(outputFilename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT),
        H5Fclose);


    for (size_t i = 0; i < namesConserved.size(); i++) {
        ASSERT_GT(H5Oexists_by_name(file.hid(), namesConserved[i].c_str(), H5P_DEFAULT),
            -1);
    }

    for (size_t i = 0; i < namesExtra.size(); i++) {
        ASSERT_GT(H5Oexists_by_name(file.hid(), namesExtra[i].c_str(), H5P_DEFAULT),
            -1);
    }


    for (size_t i = 0; i < namesConserved.size(); i++) {
        std::vector<double> data(nx * ny * nz);
        HDF5Resource dataset(H5Dopen2(file.hid(), namesConserved[i].c_str(),
                H5P_DEFAULT), H5Dclose);
        HDF5_SAFE_CALL(H5Dread(dataset.hid(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                H5P_DEFAULT,
                data.data()));

        for (size_t j = 0; j < data.size(); j++) {
            ASSERT_EQ((1 << i) + j, data[j]);
        }
    }

    for (size_t i = 0; i < namesExtra.size(); i++) {
        std::vector<double> data(nx * ny * nz);
        HDF5Resource dataset(H5Dopen2(file.hid(), namesExtra[i].c_str(), H5P_DEFAULT),
            H5Dclose);
        HDF5_SAFE_CALL(H5Dread(dataset.hid(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                H5P_DEFAULT,
                data.data()));

        for (size_t j = 0; j < data.size(); j++) {
            ASSERT_EQ((2 << i) + j, data[j]);
        }
    }

}

#endif
