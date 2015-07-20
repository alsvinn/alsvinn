#include "alsfvm/DeviceConfiguration.hpp"

namespace alsfvm {

	DeviceConfiguration::DeviceConfiguration()
        : platform("cpu")
	{
	}


	DeviceConfiguration::~DeviceConfiguration()
    {
    }

    const std::string &DeviceConfiguration::getPlatform() const
    {
        return platform;
    }

}
