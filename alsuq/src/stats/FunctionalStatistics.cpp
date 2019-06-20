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

#include "alsuq/stats/FunctionalStatistics.hpp"
#include "alsfvm/functional/FunctionalFactory.hpp"
#include "alsuq/stats/stats_util.hpp"

namespace alsuq {
namespace stats {

FunctionalStatistics::FunctionalStatistics(
    const StatisticsParameters& parameters)
    : StatisticsHelper(parameters) {

    alsfvm::functional::FunctionalFactory functionalFactory;

    platform = parameters.getPlatform();
    const std::string name = parameters.getString("functionalName");
    functional = functionalFactory.makeFunctional(platform, name,
            parameters);

    statisticsNames = {"mean_" + name};

}

std::vector<std::string> FunctionalStatistics::getStatisticsNames()
const {
    return statisticsNames;
}

void FunctionalStatistics::computeStatistics(
    const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {



    auto gridFunctional = functional->getFunctionalSize(grid);
    auto& functionalData = this->findOrCreateSnapshot(statisticsNames[0],
            timestepInformation,
            conservedVariables,
            gridFunctional.x, gridFunctional.y, gridFunctional.z, platform);





    functional->operator ()(*functionalData.getVolumes().getConservedVolume(),
        conservedVariables,
        1,
        grid
    );


}


void FunctionalStatistics::finalizeStatistics() {

}

REGISTER_STATISTICS(cuda, functional,
    FunctionalStatistics)
REGISTER_STATISTICS(cpu, functional,
    FunctionalStatistics)


}
}
