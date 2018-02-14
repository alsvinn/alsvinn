#pragma once
#include <string>
namespace alsfvm {
class DeviceConfiguration {
public:
    DeviceConfiguration(const std::string& platform = "cpu");
    virtual ~DeviceConfiguration();

    ///
    /// \brief getPlatform returns the platform the device configuration is
    ///                    for. (Either CPU or CUDA at the moment).
    /// \return "cpu" if we are running on CPU,
    ///         "cuda" if we are running on GPU/CUDA.
    ///
    const std::string& getPlatform() const;
private:
    std::string platform;
};
}
