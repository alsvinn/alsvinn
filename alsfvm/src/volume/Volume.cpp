#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
	namespace volume {
        Volume::Volume(const std::vector<std::string>& variableNames,
           std::shared_ptr<memory::MemoryFactory> memoryFactory,
           size_t nx, size_t ny, size_t nz)
            : variableNames(variableNames), memoryFactory(memoryFactory)
		{
		}


		Volume::~Volume()
		{
		}

		size_t Volume::getNumberOfVariables() const {
            return variableNames.size();
		}

	}
}
