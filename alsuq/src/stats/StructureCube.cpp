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

#include "alsuq/stats/StructureCube.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
#include "alsfvm/functional/structure_common.hpp"
namespace alsuq {
namespace stats {

StructureCube::StructureCube(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      p(parameters.getDouble("p")),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_cube_" + std::to_string(p))

{

}

std::vector<std::string> StructureCube::getStatisticsNames() const {
    return {statisticsName};
}

void StructureCube::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables,
            numberOfH, 1, 1);

    const auto boundaryConditions = grid.getBoundaryCondition(0);

    for (auto boundaryConditionOnSide : grid.getBoundaryConditions()) {
        if (boundaryConditionOnSide != boundaryConditions) {
            THROW("We require that the boundary conditions are the same on all "
                << "sides for structurere CUBE. "
                << "Given " << boundaryConditions << " and " << boundaryConditionOnSide);
        }
    }

    if (boundaryConditions == alsfvm::boundary::PERIODIC) {
        alsfvm::functional::dispatchComputeStructureCubeCPU<alsfvm::boundary::PERIODIC>(
            *structure.getVolumes().getConservedVolume(),
            conservedVariables,
            numberOfH,
            p);
    } else if (boundaryConditions == alsfvm::boundary::NEUMANN) {
        alsfvm::functional::dispatchComputeStructureCubeCPU<alsfvm::boundary::NEUMANN>(
            *structure.getVolumes().getConservedVolume(),
            conservedVariables,
            numberOfH,
            p);
    } else {
        THROW("Unsupported boundary condition for StructureCube structure functions. "
            << "Maybe you are trying to run MPI with multi_x, multi_y or multi_z > 1?"
            << " This is not supported in the current version.");
    }


}

void StructureCube::finalizeStatistics() {

}

REGISTER_STATISTICS(cpu, structure_cube, StructureCube)
}
}
