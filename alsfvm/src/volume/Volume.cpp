#include "alsfvm/volume/Volume.hpp"

namespace alsfvm {
	namespace volume {
		Volume::Volume(size_t numberOfVariables)
			: numberOfVariables(numberOfVariables)
		{
		}


		Volume::~Volume()
		{
		}

		size_t Volume::getNumberOfVariables() const {
			return numberOfVariables;
		}

	}
}