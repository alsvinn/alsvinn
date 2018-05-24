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

#include "alsuq/stats/OnePointMoment.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {

OnePointMoment::OnePointMoment(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters), p(parameters.getInteger("p")),
      statisticsName("m" + std::to_string(p)) {

}

std::vector<std::string> OnePointMoment::getStatisticsNames() const {
    return  {statisticsName};
}

void OnePointMoment::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {

    auto& m = findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables,
            extraVariables);






    m.getVolumes().getConservedVolume()->addPower(conservedVariables, p);
    m.getVolumes().getExtraVolume()->addPower(extraVariables, p);


}

void OnePointMoment::finalizeStatistics() {

}

REGISTER_STATISTICS(cpu, onepointmoment, OnePointMoment)
REGISTER_STATISTICS(cuda, onepointmoment, OnePointMoment)
}
}

