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

#include "alsuq/stats/StructureBasic.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
#include "alsfvm/boundary/ValueAtBoundary.hpp"
namespace alsuq {
namespace stats {

StructureBasic::StructureBasic(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      direction(parameters.getInteger("direction")),
      p(parameters.getDouble("p")),
      directionVector(make_direction_vector(direction)),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_basic_" + std::to_string(p))

{

}

std::vector<std::string> StructureBasic::getStatisticsNames() const {
    return {statisticsName};
}

void StructureBasic::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables,
            numberOfH, 1, 1);

    if (grid.getBoundaryCondition(direction) == alsfvm::boundary::PERIODIC) {
        computeStructure<alsfvm::boundary::PERIODIC>
        (*structure.getVolumes().getConservedVolume(),
            conservedVariables);
    } else if (grid.getBoundaryCondition(direction) == alsfvm::boundary::NEUMANN) {
        computeStructure<alsfvm::boundary::NEUMANN>
        (*structure.getVolumes().getConservedVolume(),
            conservedVariables);
    } else {
        THROW("Unsupported boundary condition for StructureBasic structure functions. "
            << "Maybe you are trying to run MPI with multi_x, multi_y or multi_z > 1?"
            << " This is not supported in the current version.");
    }
}

void StructureBasic::finalizeStatistics() {

}

template<alsfvm::boundary::Type BoundaryType>
void StructureBasic::computeStructure(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input) {
    for (size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();
        auto numberOfGhostCells = input.getNumberOfGhostCells();


        int ngx = int(input.getNumberOfXGhostCells());
        int ngy = int(input.getNumberOfYGhostCells());
        int ngz = int(input.getNumberOfZGhostCells());

        int nx = int(input.getNumberOfXCells());
        int ny = int(input.getNumberOfYCells());
        int nz = int(input.getNumberOfZCells());

        const auto numberOfCellsWithoutGhostCells = ivec3{nx, ny, nz};

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    for (int h = 0; h < int(numberOfH); ++h) {


                        const auto u_ijk = inputView.at(i + ngx, j + ngy, k + ngz);




                        const auto discretePositionPlusH = ivec3{i, j, k} + h* directionVector;

                        // For now we assume periodic boundary conditions
                        //auto u_ijk_h = inputView.at((i + h * directionVector.x) % nx + ngx,
                        //        (j + h * directionVector.y) % ny + ngy,
                        //        (k + h * directionVector.z) % nz + ngz);

                        const auto u_ijk_h =
                            alsfvm::boundary::ValueAtBoundary<BoundaryType>::getValueAtBoundary(
                                inputView,
                                discretePositionPlusH,
                                numberOfCellsWithoutGhostCells,
                                numberOfGhostCells);

                        outputView.at(h, 0, 0) += std::pow(std::abs(u_ijk - u_ijk_h),
                                p) / (nx * ny * nz);
                    }
                }
            }
        }


    }
}
REGISTER_STATISTICS(cpu, structure_basic, StructureBasic)
}
}
