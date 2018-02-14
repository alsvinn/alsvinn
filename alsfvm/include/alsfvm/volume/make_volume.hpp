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
