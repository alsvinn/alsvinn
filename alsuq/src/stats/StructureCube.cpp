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
#include "alsuq/stats/structure_common.hpp"
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
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables, extraVariables,
            numberOfH, 1, 1);


    computeStructure(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
    computeStructure(*structure.getVolumes().getExtraVolume(),
        extraVariables);
}

void StructureCube::finalizeStatistics() {

}

void StructureCube::computeStructure(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input) {
    for (size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        int ngx = input.getNumberOfXGhostCells();
        int ngy = input.getNumberOfYGhostCells();
        int ngz = input.getNumberOfZGhostCells();

        int nx = int(input.getNumberOfXCells()) - 2 * ngx;
        int ny = int(input.getNumberOfYCells()) - 2 * ngy;
        int nz = int(input.getNumberOfZCells()) - 2 * ngz;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    for (int h = 1; h < numberOfH; ++h) {

                        computeCube(outputView, inputView, i, j, k, h, nx, ny, nz,
                            ngx, ngy, ngz, input.getDimensions());

                    }
                }
            }
        }


    }
}

void StructureCube::computeCube(alsfvm::memory::View<real>& output,
    const alsfvm::memory::View<const real>& input,
    int i, int j, int k, int h, int nx, int ny, int nz,
    int ngx, int ngy, int ngz, int dimensions) {

    computeStructureCube(output, input, i, j, k, h, nx, ny, nz, ngx, ngy, ngz,
        dimensions, p);
}
REGISTER_STATISTICS(cpu, structure_cube, StructureCube)
}
}
