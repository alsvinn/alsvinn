#include "alsfvm/volume/Volume.hpp"
#include "gtest/gtest.h"

TEST(VolumeTest, SizeTest) {
    std::vector<std::string> variableNames = {"a", "b"};

    const size_t nx=10;
    const size_t ny=10;
    const size_t nz=10;

    std::shared_ptr<alsfvm::memory::MemoryFactory> factory;
    alsfvm::volume::Volume volume(variableNames, factory, nx, ny, nz);
    ASSERT_EQ(variableNames.size(), volume.getNumberOfVariables());
}
