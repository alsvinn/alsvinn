#include <gtest/gtest.h>
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"

#include "alsfvm/equation/euler/Euler.hpp"

using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::equation;
using namespace alsfvm::volume;

class VolumeForEachTest : public ::testing::Test {
public:
	std::shared_ptr<DeviceConfiguration> deviceConfiguration;
	const std::string equation;
	const std::string platform;
	const size_t nx;
	const size_t ny;
	const size_t nz;
	std::shared_ptr<MemoryFactory> memoryFactory;
	VolumeFactory volumeFactory;
	std::shared_ptr<Volume> conservedVolume;
	std::shared_ptr<Volume> extraVolume;

	VolumeForEachTest()
		: deviceConfiguration(new DeviceConfiguration("cpu")),
		equation("euler"),
		platform("cpu"),
		nx(10), ny(10), nz(10),
		memoryFactory(new MemoryFactory(deviceConfiguration)),
		volumeFactory("euler", memoryFactory),
		conservedVolume(volumeFactory.createConservedVolume(nx, ny, nz)),
		extraVolume(volumeFactory.createExtraVolume(nx, ny, nz))
	{

	}
};

TEST_F(VolumeForEachTest, EulerTestForAllIndices) {
	std::vector<size_t> indicesFound;

	// Loop through all indices, and make sure we do not have repitition
	for_each_cell_index(*conservedVolume, [&](size_t index){
		for (size_t i = 0; i < indicesFound.size(); i++) {
			ASSERT_FALSE(index == indicesFound[i]);
		}
		indicesFound.push_back(index);
	});

	ASSERT_EQ(nx*ny*nz, indicesFound.size());

	// Make sure we find all indices
	for (size_t i = 0; i < nx*ny*nz; i++) {
		bool indexFound = false;
		for (size_t j = 0; j < indicesFound.size(); j++) {
			if (indicesFound[j] == i) {
				indexFound = true;
			}
		}
		ASSERT_TRUE(indexFound);
	}
}