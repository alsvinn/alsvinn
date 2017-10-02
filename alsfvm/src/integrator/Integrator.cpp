#include "alsfvm/integrator/Integrator.hpp"

namespace alsfvm { namespace integrator { 
    real Integrator::computeTimestep(const rvec3& waveSpeeds, const rvec3& cellLengths, real cfl, const simulator::TimestepInformation& timestepInformation) const {
		real waveSpeedTotal = 0;
		for (size_t direction = 0; direction < 3; ++direction) {
			const real waveSpeed = waveSpeeds[direction];
			if (cellLengths[direction] == 0) {
				continue;
			}
			const real cellLength = cellLengths[direction];
			waveSpeedTotal += waveSpeed / cellLength;
		}

        for (auto& adjuster : waveSpeedAdjusters) {
            waveSpeedTotal = adjuster->adjustWaveSpeed(waveSpeedTotal);
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
