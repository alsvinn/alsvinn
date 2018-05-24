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

#include "alsfvm/integrator/RungeKutta2.hpp"
#include <iostream>
namespace alsfvm {
namespace integrator {

///
/// Returns the number of substeps this integrator uses.
/// Since this is second order RK, we need two subtimesteps
///
/// \returns 2
///
RungeKutta2::RungeKutta2(alsfvm::shared_ptr<System> system)
    : system(system) {

}

size_t RungeKutta2::getNumberOfSubsteps() const {
    return 2;
}


real RungeKutta2::performSubstep(
    std::vector<alsfvm::shared_ptr<volume::Volume> >& inputConserved,
    rvec3 spatialCellSizes, real dt, real cfl,
    volume::Volume& output, size_t substep,
    const simulator::TimestepInformation& timestepInformation) {
    // We compute U + dt * F(U)


    // Compute F(U)
    rvec3 waveSpeeds(0, 0, 0);
    (*system)(*inputConserved[substep], waveSpeeds, true,  output);

    if (substep == 0) {
        dt = computeTimestep(waveSpeeds, spatialCellSizes, cfl, timestepInformation);
    }

    rvec3 cellScaling(dt / spatialCellSizes.x,
        dt / spatialCellSizes.y,
        dt / spatialCellSizes.z);

    output *= cellScaling.x;
    output += *inputConserved[substep];

    if (substep == 1) {
        // 0.5 * (U+U')
        output += *inputConserved[0];
        output *= 0.5;
    }

    return dt;
}
}
}
