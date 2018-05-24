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

#include "alsuq/stats/TimeIntegratedFunctionalStatistics.hpp"
#include "alsfvm/functional/FunctionalFactory.hpp"
#include "alsuq/stats/stats_util.hpp"

namespace alsuq {
namespace stats {

TimeIntegratedFunctionalStatistics::TimeIntegratedFunctionalStatistics(
    const StatisticsParameters& parameters)
    : StatisticsHelper(parameters) {

    alsfvm::functional::FunctionalFactory functionalFactory;

    platform = parameters.getPlatform();
    const std::string name = parameters.getString("functionalName");
    functional = functionalFactory.makeFunctional(platform, name,
            parameters);

    statisticsNames = {"mean_" + name};


    time = parameters.getDouble("time");
    timeRadius = parameters.getDouble("timeRadius");


    fixedTimestepInformation = alsfvm::simulator::TimestepInformation(time, 0);

}

std::vector<std::string> TimeIntegratedFunctionalStatistics::getStatisticsNames()
const {
    return statisticsNames;
}

void TimeIntegratedFunctionalStatistics::computeStatistics(
    const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {


    const double currentTime = timestepInformation.getCurrentTime();

    auto gridFunctional = functional->getFunctionalSize(grid);
    auto& functionalTime = this->findOrCreateSnapshot(statisticsNames[0],
            fixedTimestepInformation,
            conservedVariables, extraVariables,
            gridFunctional.x, gridFunctional.y, gridFunctional.z, platform);

    if (std::abs(currentTime - time) <= timeRadius) {
        // Now we should write
        double dt = currentTime - lastTime;

        // If dt <= 0, we get no contributioni (usually it is because we
        // are on a new sample)
        if (dt > 0) {
            functional->operator ()(*functionalTime.getVolumes().getConservedVolume(),
                *functionalTime.getVolumes().getExtraVolume(),
                conservedVariables,
                extraVariables,
                dt,
                grid
            );
        }

    }

    lastTime = currentTime;


}

void TimeIntegratedFunctionalStatistics::finalizeStatistics() {

}

REGISTER_STATISTICS(cuda, functional_time_integrated,
    TimeIntegratedFunctionalStatistics)
REGISTER_STATISTICS(cpu, functional_time_integrated,
    TimeIntegratedFunctionalStatistics)


}
}
