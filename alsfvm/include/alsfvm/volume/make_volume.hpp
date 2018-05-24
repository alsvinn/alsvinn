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

#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"

namespace alsfvm {
namespace volume {
//! Convenience function meant to be used for testing, NOT in production
inline alsfvm::shared_ptr<Volume> makeConservedVolume(const std::string&
    platform,
    const std::string& equation,
    const ivec3& innerSize,
    const int ghostCells) {

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration(platform));

    auto memoryFactory = alsfvm::make_shared<memory::MemoryFactory>
        (deviceConfiguration);

    VolumeFactory factory(equation, memoryFactory);

    return factory.createConservedVolume(innerSize.x, innerSize.y, innerSize.z,
            ghostCells);
}

//! Convenience function meant to be used for testing, NOT in production
inline alsfvm::shared_ptr<Volume> makeExtraVolume(const std::string& platform,
    const std::string& equation,
    const ivec3& innerSize,
    const int ghostCells) {

    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration(
        new DeviceConfiguration(platform));

    auto memoryFactory = alsfvm::make_shared<memory::MemoryFactory>
        (deviceConfiguration);

    VolumeFactory factory(equation, memoryFactory);

    return factory.createExtraVolume(innerSize.x, innerSize.y, innerSize.z,
            ghostCells);
}
}
}
