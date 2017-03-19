#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/mpi/utils.hpp"
#include "alsutils/mpi/cuda.hpp"
#include "alsutils/log.hpp"

namespace alsuq { namespace stats { 
StatisticsHelper::StatisticsHelper(const StatisticsParameters &parameters)

    : samples(parameters.getNumberOfSamples())
{

}

void StatisticsHelper::addWriter(const std::string& name,
                                 std::shared_ptr<alsfvm::io::Writer> &writer)
{
    writers[name].push_back(writer);
}

void StatisticsHelper::combineStatistics()
{

    for (auto& snapshot : snapshots) {
        for(auto& statistics : snapshot.second) {
            for (auto& volume: statistics.second.getVolumes()) {
                for (size_t variable = 0; variable < volume->getNumberOfVariables();
                     variable++) {

                    auto statisticsData = volume->getScalarMemoryArea(variable);
                    auto statisticsDataToReduce = statisticsData;


                    // Check if we should copy to CPU
                    if (!statisticsData->isOnHost() && !alsutils::mpi::hasGPUDirectSupport()) {

                        statisticsDataToReduce = statisticsData->getHostMemory();

                        ALSVINN_LOG(INFO, "Copying from GPU, now statisticsDataToReduce.isOnHost() = " << statisticsDataToReduce->isOnHost() );

                    }


                    std::shared_ptr<alsfvm::memory::Memory<real> > dataReduced;
                    //if (mpiConfig.getRank() == 0) {
                        dataReduced = statisticsDataToReduce->makeInstance();
                    //}
                    MPI_SAFE_CALL(MPI_Reduce(statisticsDataToReduce->data(), dataReduced->data(),
                                             statisticsDataToReduce->getSize(), MPI_DOUBLE, MPI_SUM, 0,
                                             mpiConfig.getCommunicator()));

                    if (mpiConfig.getRank() == 0) {
                        statisticsData->copyFrom(*dataReduced);
                    }

                    *statisticsData /= samples;
                }
            }
        }
    }
}

void StatisticsHelper::writeStatistics(const alsfvm::grid::Grid &grid)
{
    for (auto& snapshot : snapshots) {
        for (auto& statistics : snapshot.second) {
            const auto& statisticsName = statistics.first;
            for(auto& writer : writers[statisticsName]) {
                auto& volumes = statistics.second.getVolumes();
                auto& timestepInformation = statistics.second.getTimestepInformation();
                writer->write(*volumes.getConservedVolume(),
                              *volumes.getExtraVolume(),
                              grid, timestepInformation);
            }
        }
    }
}

StatisticsSnapshot &StatisticsHelper::findOrCreateSnapshot(
        const std::string& name,
        const alsfvm::simulator::TimestepInformation &timestepInformation,
        const alsfvm::volume::Volume &conservedVariables,
        const alsfvm::volume::Volume &extraVariables)
{
    auto currentTime = timestepInformation.getCurrentTime();
    if (snapshots.find(currentTime) != snapshots.end()
            && snapshots[currentTime].find(name) != snapshots[currentTime].end()) {
        return snapshots[currentTime][name];
    } else {
        auto conservedVariablesClone = conservedVariables.makeInstance();
        auto extraVariablesClone = extraVariables.makeInstance();
        snapshots[currentTime][name] = StatisticsSnapshot(timestepInformation,
                                                          alsfvm::volume::VolumePair(conservedVariablesClone,
                                                                                     extraVariablesClone));

        for (auto& volume : snapshots[currentTime][name].getVolumes()) {
            volume->makeZero();
        }

        return snapshots[currentTime][name];

    }

}

StatisticsSnapshot &StatisticsHelper::findOrCreateSnapshot(const std::string& name,
                                                           const alsfvm::simulator::TimestepInformation &timestepInformation,
                                                           const alsfvm::volume::Volume &conservedVariables,
                                                           const alsfvm::volume::Volume &extraVariables,
                                                           size_t nx, size_t ny, size_t nz)
{
    auto currentTime = timestepInformation.getCurrentTime();
    if (snapshots.find(currentTime) != snapshots.end()
            && snapshots[currentTime].find(name) != snapshots[currentTime].end()) {
        return snapshots[currentTime][name];
    } else {
        auto conservedVariablesClone = conservedVariables.makeInstance(nx, ny, nz);
        auto extraVariablesClone = extraVariables.makeInstance(nx, ny, nz);
        snapshots[currentTime][name] = StatisticsSnapshot(timestepInformation,
                                                          alsfvm::volume::VolumePair(conservedVariablesClone,
                                                                                     extraVariablesClone));

        for (auto& volume : snapshots[currentTime][name].getVolumes()) {
            volume->makeZero();
        }

        return snapshots[currentTime][name];

    }
}
}
                }
