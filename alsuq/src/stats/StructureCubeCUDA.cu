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

#include "alsuq/stats/StructureCubeCUDA.hpp"

#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
#include "alsuq/stats/structure_common.hpp"
#include "alsutils/math/FastPower.hpp"
#include "alsutils/math/PowPower.hpp"
#include "alsfvm/functional/structure_common_cuda.hpp"
namespace alsuq {
namespace stats {

namespace {





//! Computes the structure function for FIXED h
//!
//! The goal is to compute the structure function, then reduce (sum) over space
//! then go on to next h
//!
template<class PowerClass>
__global__ void computeStructureCube(real* output,
    alsfvm::memory::View<const real> input,
    int h,
    int nx, int ny, int nz, int ngx, int ngy, int ngz,
    real p, int dimensions) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nx * ny * nz) {
        return;
    }

    const int i = index % nx;
    const int j = (index / nx) % ny;
    const int k = (index / nx / ny);


    output[index] = 0;
    forEachPointInComputeStructureCube([&] (double u, double u_h) {
        output[index] += PowerClass::power(fabs(u - u_h), p);
    }, input, i, j, k, h, nx, ny, nz, ngx, ngy, ngz, dimensions);

}

}

StructureCubeCUDA::StructureCubeCUDA(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      p(parameters.getDouble("p")),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_cube_" + std::to_string(p))

{

}

std::vector<std::string> StructureCubeCUDA::getStatisticsNames() const {
    return {statisticsName};
}

void StructureCubeCUDA::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables,
            numberOfH, 1, 1, "cpu");

    alsfvm::functional::dispatchComputeStructureCubeCUDA(*structure.getVolumes().getConservedVolume(),
                                              conservedVariables, structureOutput, numberOfH, p);

}

void StructureCubeCUDA::finalizeStatistics() {

}

REGISTER_STATISTICS(cuda, structure_cube, StructureCubeCUDA)
}
}
