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

#include "alsuq/stats/FixedIntervalStatistics.hpp"
#include "alsutils/log.hpp"

namespace alsuq {
namespace stats {

FixedIntervalStatistics::FixedIntervalStatistics(alsfvm::shared_ptr<Statistics>&
    statistics, real timeInterval, real endTime)
    : statistics(statistics), timeInterval(timeInterval), endTime(endTime) {

}

real FixedIntervalStatistics::adjustTimestep(real dt,
    const alsfvm::simulator::TimestepInformation& timestepInformation) const {
    if (numberSaved > 0) {
        const real nextSaveTime = numberSaved * timeInterval;
        return std::min(dt, nextSaveTime - timestepInformation.getCurrentTime());
    } else {
        return dt;
    }
}

void FixedIntervalStatistics::combineStatistics() {
    statistics->combineStatistics();
}

void FixedIntervalStatistics::addWriter(const std::string& name,
    std::shared_ptr<alsfvm::io::Writer>& writer) {
    statistics->addWriter(name, writer);
}

std::vector<std::string> FixedIntervalStatistics::getStatisticsNames() const {
    return statistics->getStatisticsNames();
}

void FixedIntervalStatistics::writeStatistics(const alsfvm::grid::Grid& grid) {
    statistics->writeStatistics(grid);
}

void FixedIntervalStatistics::finalizeStatistics() {
    statistics->finalizeStatistics();
}

void FixedIntervalStatistics::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {


    const real currentTime = timestepInformation.getCurrentTime();

    // First check if we have restarted
    if (currentTime == 0) {
        numberSaved = 0;
    }

    if (currentTime >= numberSaved * timeInterval) {
        ALSVINN_LOG(INFO, "Computing statistics, currentTime = " << currentTime << ", "
            << "\n\tnumberSaves = " << numberSaved
            << "\n\ttimeInterval = " << timeInterval
            << "\n\tendTime = " << endTime);
        statistics->computeStatistics(conservedVariables, extraVariables, grid,
            timestepInformation);
        numberSaved++;
    }
}

}
}
