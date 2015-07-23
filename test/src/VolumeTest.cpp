#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/memory/HostMemory.hpp"
#include "gtest/gtest.h"
#include "alsfvm/volume/VolumeFactory.hpp"

using namespace alsfvm::volume;

TEST(VolumeTest, SizeTest) {
    std::vector<std::string> variableNames = {"a", "b"};

    const size_t nx=10;
    const size_t ny=10;
    const size_t nz=10;

    auto configuration = std::make_shared<alsfvm::DeviceConfiguration>();
    auto factory = std::make_shared<alsfvm::memory::MemoryFactory>("HostMemory",
                                                                   configuration);
    alsfvm::volume::Volume volume(variableNames, factory, nx, ny, nz);
    ASSERT_EQ(variableNames.size(), volume.getNumberOfVariables());
}

TEST(VolumeTest, GetVariableIndex) {
	std::vector<std::string> variableNames = { "alpha", "beta" };

	const size_t nx = 10;
	const size_t ny = 10;
	const size_t nz = 10;

    auto configuration = std::make_shared<alsfvm::DeviceConfiguration>();

    auto factory = std::make_shared<alsfvm::memory::MemoryFactory>("HostMemory",
                                                                   configuration);
	alsfvm::volume::Volume volume(variableNames, factory, nx, ny, nz);

	ASSERT_EQ(0, volume.getIndexFromName("alpha"));
	ASSERT_EQ(1, volume.getIndexFromName("beta"));
	ASSERT_EQ(std::string("alpha"), volume.getName(0));
	ASSERT_EQ(std::string("beta"), volume.getName(1));

}

TEST(VolumeTest, WriteToMemoryArea) {
    std::vector<std::string> variableNames = { "alpha", "beta" };

    const size_t nx = 10;
    const size_t ny = 10;
    const size_t nz = 10;

    auto configuration = std::make_shared<alsfvm::DeviceConfiguration>();

    auto factory = std::make_shared<alsfvm::memory::MemoryFactory>("HostMemory",
                                                                   configuration);
    alsfvm::volume::Volume volume(variableNames, factory, nx, ny, nz);

    auto memory0 = volume.getScalarMemoryArea("alpha");

    ASSERT_EQ(nx, memory0->getSizeX());
    ASSERT_EQ(ny, memory0->getSizeY());
    ASSERT_EQ(nz, memory0->getSizeZ());

    std::dynamic_pointer_cast<alsfvm::memory::HostMemory<alsfvm::real> >(memory0)->at(0,0,0) = 10;


}

TEST(VolumeTest, FactoryTestEuler) {
	const size_t nx = 10;
	const size_t ny = 10;
	const size_t nz = 10;

	auto configuration = std::make_shared<alsfvm::DeviceConfiguration>();

	auto factory = std::make_shared<alsfvm::memory::MemoryFactory>("HostMemory",
		configuration);

	VolumeFactory volumeFactory("euler", factory);

	auto eulerConserved = volumeFactory.createConservedVolume(nx, ny, nz);

	ASSERT_EQ(5, eulerConserved->getNumberOfVariables());

	ASSERT_EQ(nx, eulerConserved->getNumberOfXCells());

	ASSERT_EQ(ny, eulerConserved->getNumberOfYCells());
	ASSERT_EQ(nz, eulerConserved->getNumberOfZCells());

	ASSERT_EQ(0, eulerConserved->getIndexFromName("rho"));
	ASSERT_EQ(1, eulerConserved->getIndexFromName("mx"));
	ASSERT_EQ(2, eulerConserved->getIndexFromName("my"));
	ASSERT_EQ(3, eulerConserved->getIndexFromName("mz"));
	ASSERT_EQ(4, eulerConserved->getIndexFromName("E"));

	auto eulerExtra = volumeFactory.createExtraVolume(nx, ny, nz);

	ASSERT_EQ(4, eulerExtra->getNumberOfVariables());

	ASSERT_EQ(nx, eulerExtra->getNumberOfXCells());

	ASSERT_EQ(ny, eulerExtra->getNumberOfYCells());
	ASSERT_EQ(nz, eulerExtra->getNumberOfZCells());

	ASSERT_EQ(0, eulerExtra->getIndexFromName("p"));
	ASSERT_EQ(1, eulerExtra->getIndexFromName("ux"));
	ASSERT_EQ(2, eulerExtra->getIndexFromName("uy"));
	ASSERT_EQ(3, eulerExtra->getIndexFromName("uz"));
	
}
