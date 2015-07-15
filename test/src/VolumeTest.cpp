#include "alsfvm/volume/Volume.hpp"
#include "gtest/gtest.h"

TEST(VolumeTest, SizeTest) {
	size_t numberOfVariables = 10;
	alsfvm::volume::Volume volume(numberOfVariables);
	ASSERT_EQ(numberOfVariables, volume.getNumberOfVariables());
}