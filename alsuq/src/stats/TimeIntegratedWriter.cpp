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

#include "alsuq/stats/TimeIntegratedWriter.hpp"


#include "alsutils/log.hpp"

namespace alsuq {
namespace stats {

TimeIntegratedWriter::TimeIntegratedWriter(alsfvm::shared_ptr<Statistics>&
    statistics,
    real time, real radius)
    : statistics(statistics) {

}


void TimeIntegratedWriter::combineStatistics() {
    statistics->combineStatistics();
}

void TimeIntegratedWriter::addWriter(const std::string& name,
    std::shared_ptr<alsfvm::io::Writer>& writer) {
    statistics->addWriter(name, writer);
}

std::vector<std::string> TimeIntegratedWriter::getStatisticsNames() const {
    return statistics->getStatisticsNames();
}

void TimeIntegratedWriter::writeStatistics(const alsfvm::grid::Grid& grid) {
    statistics->writeStatistics(grid);
}

void TimeIntegratedWriter::finalizeStatistics() {
    statistics->finalizeStatistics();
}

void TimeIntegratedWriter::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {




    statistics->computeStatistics(conservedVariables, grid,
        timestepInformation);

}

}
}
