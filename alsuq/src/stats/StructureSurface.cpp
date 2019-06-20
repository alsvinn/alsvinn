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

#include "alsuq/stats/StructureSurface.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {

StructureSurface::StructureSurface(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      p(parameters.getDouble("p")),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_surface_" + std::to_string(p))

{

}

std::vector<std::string> StructureSurface::getStatisticsNames() const {
    return {statisticsName};
}

void StructureSurface::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables,
            numberOfH, 1, 1);


    computeStructure(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
}

void StructureSurface::finalizeStatistics() {

}

void StructureSurface::computeStructure(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input) {
    THROW("Not implemented yet");
}
REGISTER_STATISTICS(cpu, structure_surface, StructureSurface)
}
}
