#pragma once
#include "alsfvm/types.hpp"
#include <string>
#include <vector>

namespace alsfvm {
	namespace volume {
		class Volume
		{
		public:
			Volume(size_t numberOfVariables);
			virtual ~Volume();

			size_t getNumberOfVariables() const;

		private:
			const size_t numberOfVariables;
		};
	}
}
