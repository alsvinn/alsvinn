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

#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/mpi/utils.hpp"
#include "alsutils/mpi/cuda.hpp"
#include "alsutils/log.hpp"
#include "alsutils/mpi/mpi_types.hpp"
namespace alsuq {
namespace stats {
StatisticsHelper::StatisticsHelper(const StatisticsParameters& parameters)

    : samples(parameters.getNumberOfSamples()),
      mpiConfig(parameters.getMpiConfiguration()) {

}

void StatisticsHelper::addWriter(const std::string& name,
    std::shared_ptr<alsfvm::io::Writer>& writer) {
    writers[name].push_back(writer);
}

void StatisticsHelper::combineStatistics() {

    for (auto& snapshot : snapshots) {
        for (auto& statistics : snapshot.second) {
            for (auto& volume : statistics.second.getVolumes()) {

                for (size_t variable = 0; variable < volume->getNumberOfVariables();
                    variable++) {



                    auto statisticsData = volume->getScalarMemoryArea(variable);
                    auto statisticsDataToReduce = statisticsData;


                    // Check if we should copy to CPU
                    if (!statisticsData->isOnHost() && !alsutils::mpi::hasGPUDirectSupport()) {

                        statisticsDataToReduce = statisticsData->getHostMemory();

                        ALSVINN_LOG(INFO, "Copying from GPU, now statisticsDataToReduce.isOnHost() = "
                            << statisticsDataToReduce->isOnHost() );

                    }


                    std::shared_ptr<alsfvm::memory::Memory<real> > dataReduced;
                    //if (mpiConfig.getRank() == 0) {
                    dataReduced = statisticsDataToReduce->makeInstance();
                    //}
                    MPI_SAFE_CALL(MPI_Reduce(statisticsDataToReduce->data(), dataReduced->data(),
                            statisticsDataToReduce->getSize(), alsutils::mpi::MpiTypes<real>::MPI_Real,
                            MPI_SUM, 0,
                            mpiConfig->getCommunicator()));

                    if (mpiConfig->getRank() == 0) {
                        statisticsData->copyFrom(*dataReduced);
                    }

                    *statisticsData /= samples;
                }
            }
        }
    }
}

void StatisticsHelper::writeStatistics(const alsfvm::grid::Grid& grid) {
    for (auto& snapshot : snapshots) {
        for (auto& statistics : snapshot.second) {
            const auto& statisticsName = statistics.first;

            for (auto& writer : writers[statisticsName]) {
                auto& volumes = statistics.second.getVolumes();
                auto& timestepInformation = statistics.second.getTimestepInformation();

                if (ownGrid) {
                    writer->write(*volumes.getConservedVolume(),
                        *ownGrid, timestepInformation);
                } else {
                    writer->write(*volumes.getConservedVolume(),
                        grid, timestepInformation);
                }
            }
        }
    }
}

StatisticsSnapshot& StatisticsHelper::findOrCreateSnapshot(
    const std::string& name,
    const alsfvm::simulator::TimestepInformation& timestepInformation,
    const alsfvm::volume::Volume& conservedVariables) {
    auto currentTime = timestepInformation.getCurrentTime();

    if (snapshots.find(currentTime) != snapshots.end()
        && snapshots[currentTime].find(name) != snapshots[currentTime].end()) {
        return snapshots[currentTime][name];
    } else {
        auto conservedVariablesClone = conservedVariables.makeInstance();

        snapshots[currentTime][name] = StatisticsSnapshot(timestepInformation,
                alsfvm::volume::VolumePair(conservedVariablesClone));

        for (auto& volume : snapshots[currentTime][name].getVolumes()) {
            volume->makeZero();
        }

        return snapshots[currentTime][name];

    }

}

StatisticsSnapshot& StatisticsHelper::findOrCreateSnapshot(
    const std::string& name,
    const alsfvm::simulator::TimestepInformation& timestepInformation,
    const alsfvm::volume::Volume& conservedVariables,
    size_t nx, size_t ny, size_t nz, const std::string& platform) {
    auto currentTime = timestepInformation.getCurrentTime();


    if (snapshots.find(currentTime) != snapshots.end()
        && snapshots[currentTime].find(name) != snapshots[currentTime].end()) {
        return snapshots[currentTime][name];
    } else {

        if (nx != conservedVariables.getNumberOfXCells()
            || ny != conservedVariables.getNumberOfYCells() ||
            nz != conservedVariables.getNumberOfZCells()) {

            ALSVINN_LOG(INFO, "Making new grid for saving statistics");
            makeOwnGrid(nx, ny, nz);
        }

        auto conservedVariablesClone = conservedVariables.makeInstance(nx, ny, nz,
                platform);

        snapshots[currentTime][name] = StatisticsSnapshot(timestepInformation,
                alsfvm::volume::VolumePair(conservedVariablesClone));

        for (auto& volume : snapshots[currentTime][name].getVolumes()) {
            volume->makeZero();
        }

        return snapshots[currentTime][name];

    }
}

void StatisticsHelper::makeOwnGrid(size_t nx, size_t ny, size_t nz) {
    ownGrid.reset(new alsfvm::grid::Grid(rvec3{0, 0, 0},
            rvec3{1, 1, 1},
            ivec3{int(nz), int(ny), int(nz)}));
}
}
}
