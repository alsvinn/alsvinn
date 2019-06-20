/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    const std::string equation;
    const std::string platform;
    const size_t nx;
    const size_t ny;
    const size_t nz;
    alsfvm::shared_ptr<MemoryFactory> memoryFactory;
    VolumeFactory volumeFactory;
    alsfvm::shared_ptr<Volume> conservedVolume;
    alsfvm::shared_ptr<Volume> extraVolume;

    VolumeForEachTest()
        : deviceConfiguration(new DeviceConfiguration("cpu")),
          equation("euler3"),
          platform("cpu"),
          nx(10), ny(10), nz(10),
          memoryFactory(new MemoryFactory(deviceConfiguration)),
          volumeFactory("euler3", memoryFactory),
          conservedVolume(volumeFactory.createConservedVolume(nx, ny, nz)),
          extraVolume(volumeFactory.createExtraVolume(nx, ny, nz)) {

    }
};

TEST_F(VolumeForEachTest, EulerTestForAllIndices) {
    std::vector<size_t> indicesFound;

    // Loop through all indices, and make sure we do not have repitition
    for_each_cell_index(*conservedVolume, [&](size_t index) {
        for (size_t i = 0; i < indicesFound.size(); i++) {
            ASSERT_FALSE(index == indicesFound[i]);
        }

        indicesFound.push_back(index);
    });

    ASSERT_EQ(nx * ny * nz, indicesFound.size());

    // Make sure we find all indices
    for (size_t i = 0; i < nx * ny * nz; i++) {
        bool indexFound = false;

        for (size_t j = 0; j < indicesFound.size(); j++) {
            if (indicesFound[j] == i) {
                indexFound = true;
            }
        }

        ASSERT_TRUE(indexFound);
    }
}