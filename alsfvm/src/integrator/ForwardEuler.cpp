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

#include "alsfvm/integrator/ForwardEuler.hpp"
#include <iostream>
namespace alsfvm {
namespace integrator {

ForwardEuler::ForwardEuler(alsfvm::shared_ptr<System> system)
    : system(system) {
    // Empty
}


size_t ForwardEuler::getNumberOfSubsteps() const {
    return 1;
}


real ForwardEuler::performSubstep(
    std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
    rvec3 spatialCellSizes, real dt, real cfl,
    volume::Volume& output, size_t substep,
    const simulator::TimestepInformation& timestepInformation) {

    rvec3 waveSpeed(0, 0, 0);

    (*system)(*inputConserved[0], waveSpeed, true, output);
    dt = computeTimestep(waveSpeed, spatialCellSizes, cfl, timestepInformation);
    rvec3 cellScaling(dt / spatialCellSizes.x,
        dt / spatialCellSizes.y,
        dt / spatialCellSizes.z);

    output *= cellScaling.x;
    output += *inputConserved[0];

    return dt;

}
}
}
