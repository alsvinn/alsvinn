#include "alsfvm/integrator/Integrator.hpp"

namespace alsfvm { namespace integrator { 
    real Integrator::computeTimestep(const rvec3& waveSpeeds, const rvec3& cellLengths, real cfl, const simulator::TimestepInformation& timestepInformation) const {
		real waveSpeedTotal = 0;
		for (size_t direction = 0; direction < 3; ++direction) {
            real waveSpeed = waveSpeeds[direction];

			if (cellLengths[direction] == 0) {
				continue;
			}

            for (auto& adjuster : waveSpeedAdjusters) {
                waveSpeed = adjuster->adjustWaveSpeed(waveSpeed);
            }
			const real cellLength = cellLengths[direction];
			waveSpeedTotal += waveSpeed / cellLength;
		}




		const real dt = cfl / waveSpeedTotal;

        return adjustTimestep(dt, timestepInformation);
    }

    void Integrator::addTimestepAdjuster(alsfvm::shared_ptr<TimestepAdjuster> &adjuster)
    {
        timestepAdjusters.push_back(adjuster);
    }

    void Integrator::addWaveSpeedAdjuster(WaveSpeedAdjusterPtr adjuster)
    {
        waveSpeedAdjusters.push_back(adjuster);
    }

    real Integrator::adjustTimestep(real dt, const simulator::TimestepInformation &timestepInformation) const
    {
        real newDt = dt;
        for(auto adjuster : timestepAdjusters) {
            newDt = adjuster->adjustTimestep(newDt, timestepInformation);
        }
        return newDt;
    }
}
}
