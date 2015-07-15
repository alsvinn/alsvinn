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

TEST(VolumeTest, GetVariableIndex) {
	std::vector<std::string> variableNames = { "alpha", "beta" };

	const size_t nx = 10;
	const size_t ny = 10;
	const size_t nz = 10;

	std::shared_ptr<alsfvm::memory::MemoryFactory> factory;
	alsfvm::volume::Volume volume(variableNames, factory, nx, ny, nz);

	ASSERT_EQ(0, volume.getIndexFromName("alpha"));
	ASSERT_EQ(1, volume.getIndexFromName("beta"));
	ASSERT_EQ(std::string("alpha"), volume.getName(0));
	ASSERT_EQ(std::string("beta"), volume.getName(1));
	
}
