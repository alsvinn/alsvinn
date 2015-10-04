#include "alsfvm/integrator/Integrator.hpp"

namespace alsfvm { namespace integrator { 
	real Integrator::computeTimestep(const rvec3& waveSpeeds, const rvec3& cellLengths, real cfl) const {
		real waveSpeedTotal = 0;
		for (size_t direction = 0; direction < 3; ++direction) {
			const real waveSpeed = waveSpeeds[direction];
			if (cellLengths[direction] == 0) {
				continue;
			}
			const real cellLength = cellLengths[direction];
			waveSpeedTotal += waveSpeed / cellLength;
		}



		const real dt = cfl / waveSpeedTotal;

		return dt;
	}
}
}
