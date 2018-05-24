/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsfvm/integrator/Integrator.hpp"

namespace alsfvm {
namespace integrator {
real Integrator::computeTimestep(const rvec3& waveSpeeds,
    const rvec3& cellLengths, real cfl,
    const simulator::TimestepInformation& timestepInformation) const {
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

void Integrator::addTimestepAdjuster(alsfvm::shared_ptr<TimestepAdjuster>&
    adjuster) {
    timestepAdjusters.push_back(adjuster);
}

void Integrator::addWaveSpeedAdjuster(WaveSpeedAdjusterPtr adjuster) {
    waveSpeedAdjusters.push_back(adjuster);
}

real Integrator::adjustTimestep(real dt,
    const simulator::TimestepInformation& timestepInformation) const {
    real newDt = dt;

    for (auto adjuster : timestepAdjusters) {
        newDt = adjuster->adjustTimestep(newDt, timestepInformation);
    }

    return newDt;
}
}
}
