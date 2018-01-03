#include "alsuq/stats/TimeIntegratedWriter.hpp"


#include "alsutils/log.hpp"

namespace alsuq { namespace stats {

TimeIntegratedWriter::TimeIntegratedWriter(alsfvm::shared_ptr<Statistics> &statistics,
                                           real time, real radius)
    : statistics(statistics), time(time), timeRadius(radius)
{

}


void TimeIntegratedWriter::combineStatistics()
{
    statistics->combineStatistics();
}

void TimeIntegratedWriter::addWriter(const std::string &name, std::shared_ptr<alsfvm::io::Writer> &writer)
{
    statistics->addWriter(name, writer);
}

std::vector<std::string> TimeIntegratedWriter::getStatisticsNames() const
{
    return statistics->getStatisticsNames();
}

void TimeIntegratedWriter::writeStatistics(const alsfvm::grid::Grid &grid)
{
    statistics->writeStatistics(grid);
}

void TimeIntegratedWriter::finalize()
{
    statistics->finalize();
}

void TimeIntegratedWriter::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                                const alsfvm::volume::Volume &extraVariables,
                                                const alsfvm::grid::Grid &grid,
                                                const alsfvm::simulator::TimestepInformation &timestepInformation)
{


    const real currentTime = timestepInformation.getCurrentTime();

    if (std::abs(currentTime-time) <= timeRadius) {

        // Here we want the statistics to "think" it is the time at which we have the interval centered,
        // so that they will accumulate the statistics on the same time.
        alsfvm::simulator::TimestepInformation timestepInformationAtTime(time,
                                                                         timestepInformation.getNumberOfStepsPerformed());

        ALSVINN_LOG(INFO, "Computing statistics, currentTime = " << currentTime);
        statistics->computeStatistics(conservedVariables, extraVariables, grid, timestepInformationAtTime);

    }
}

}
}
