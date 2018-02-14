#include "alsfvm/DeviceConfiguration.hpp"

namespace alsfvm {

DeviceConfiguration::DeviceConfiguration(const std::string& platform)
    : platform(platform) {
}


DeviceConfiguration::~DeviceConfiguration() {
}

const std::string& DeviceConfiguration::getPlatform() const {
    return platform;
}

}
