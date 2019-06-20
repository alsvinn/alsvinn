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

#include "alsuq/stats/BoundedVariation.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {



BoundedVariation::BoundedVariation(const alsuq::stats::StatisticsParameters&
    parameters)
    : StatisticsHelper(parameters), p(parameters.getInteger("p")),
      statisticsName("bv_" + std::to_string(p)) {

}

std::vector<std::string> BoundedVariation::getStatisticsNames() const {
    return {statisticsName};
}

void BoundedVariation::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& bv = findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables, 1, 1, 1, "cpu");

    const ivec3 start = conservedVariables.getNumberOfGhostCells();
    const ivec3 end = conservedVariables.getSize() -
        conservedVariables.getNumberOfGhostCells();

    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        bv.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0]
            = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(p,
                    start,
                    end);
    }
}

void BoundedVariation::finalizeStatistics() {

}
REGISTER_STATISTICS(cpu, bv, BoundedVariation)
REGISTER_STATISTICS(cuda, bv, BoundedVariation)
}
}
