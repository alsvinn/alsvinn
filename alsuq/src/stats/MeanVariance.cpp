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

#include "alsuq/stats/MeanVariance.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {

MeanVariance::MeanVariance(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters) {

}

std::vector<std::string> MeanVariance::getStatisticsNames() const {
    return  {"mean", "variance"};
}

void MeanVariance::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {

    auto& mean = findOrCreateSnapshot("mean",
            timestepInformation,
            conservedVariables
        );

    auto& variance = findOrCreateSnapshot("variance",
            timestepInformation,
            conservedVariables);



    *mean.getVolumes().getConservedVolume() += conservedVariables;


    variance.getVolumes().getConservedVolume()->addPower(conservedVariables, 2);



}

void MeanVariance::finalizeStatistics() {
    for (auto& snapshot : this->snapshots) {
        auto& secondMoment = snapshot.second["variance"];
        auto& volumesMoment = secondMoment.getVolumes();

        auto& mean = snapshot.second["mean"];
        auto& volumesMean = mean.getVolumes();

        volumesMoment.getConservedVolume()->subtractPower(
            *volumesMean.getConservedVolume(), 2.);

    }
}
REGISTER_STATISTICS(cpu, meanvar, MeanVariance)
REGISTER_STATISTICS(cuda, meanvar, MeanVariance)
}
}

