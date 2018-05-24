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

#include "alsuq/stats/BoundedVariationDirection.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {



BoundedVariationDirection::BoundedVariationDirection(const
    alsuq::stats::StatisticsParameters& parameters)
    : StatisticsHelper(parameters), p(parameters.getInteger("p")),
      statisticsNames({"bv_x_" + std::to_string(p),
    "bv_y_" + std::to_string(p),
    "bv_z_" + std::to_string(p)}) {

}

std::vector<std::string> BoundedVariationDirection::getStatisticsNames() const {
    return statisticsNames;
}

void BoundedVariationDirection::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    const ivec3 start = conservedVariables.getNumberOfGhostCells();
    const ivec3 end = conservedVariables.getSize() -
        conservedVariables.getNumberOfGhostCells();

    auto& bvX = findOrCreateSnapshot(statisticsNames[0],
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    auto& bvY = findOrCreateSnapshot(statisticsNames[1],
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    auto& bvZ = findOrCreateSnapshot(statisticsNames[2],
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        bvX.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0]
            = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(0, p, start,
                    end);

        if (conservedVariables.getDimensions() > 1) {
            bvY.getVolumes().getConservedVolume()->getScalarMemoryArea(
                var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(
                        var)->getTotalVariation(1, p, start, end);
        }

        if (conservedVariables.getDimensions() > 2) {
            bvZ.getVolumes().getConservedVolume()->getScalarMemoryArea(
                var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(
                        var)->getTotalVariation(2, p, start, end);
        }

    }

    for (size_t var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        bvX.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] =
            conservedVariables.getScalarMemoryArea(var)->getTotalVariation(0, p, start,
                end);

        if (conservedVariables.getDimensions() > 1) {
            bvY.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] =
                conservedVariables.getScalarMemoryArea(var)->getTotalVariation(1, p, start,
                    end);
        }

        if (conservedVariables.getDimensions() > 2) {
            bvZ.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] =
                conservedVariables.getScalarMemoryArea(var)->getTotalVariation(2, p, start,
                    end);
        }
    }







}

void BoundedVariationDirection::finalizeStatistics() {

}
REGISTER_STATISTICS(cpu, bv_direction, BoundedVariationDirection)
REGISTER_STATISTICS(cuda, bv_direction, BoundedVariationDirection)
}
}

