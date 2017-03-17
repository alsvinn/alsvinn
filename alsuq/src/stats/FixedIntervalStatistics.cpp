#include "alsuq/stats/FixedIntervalStatistics.hpp"

namespace alsuq { namespace stats {

FixedIntervalStatistics::FixedIntervalStatistics(alsfvm::shared_ptr<Statistics> &statistics, real timeInterval, real endTime)
    : statistics(statistics), timeInterval(timeInterval), endTime(endTime)
{

}

real FixedIntervalStatistics::adjustTimestep(real dt, const alsfvm::simulator::TimestepInformation &timestepInformation) const
{
    if (numberSaved > 0) {
        const real nextSaveTime = numberSaved * timeInterval;
        return std::min(dt, nextSaveTime - timestepInformation.getCurrentTime());
    } else {
        return dt;
    }
}

void FixedIntervalStatistics::combineStatistics()
{
    statistics->combineStatistics();
}

void FixedIntervalStatistics::addWriter(const std::string &name, std::shared_ptr<alsfvm::io::Writer> &writer)
{
    statistics->addWriter(name, writer);
}

std::vector<std::string> FixedIntervalStatistics::getStatisticsNames() const
{
    return statistics->getStatisticsNames();
}

void FixedIntervalStatistics::writeStatistics(const alsfvm::grid::Grid &grid)
{
    statistics->writeStatistics(grid);
}

void FixedIntervalStatistics::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                                const alsfvm::volume::Volume &extraVariables,
                                                const alsfvm::grid::Grid &grid,
                                                const alsfvm::simulator::TimestepInformation &timestepInformation)
{
    const real currentTime = timestepInformation.getCurrentTime();
    if (currentTime >= numberSaved * timeInterval) {
        statistics->computeStatistics(conservedVariables, extraVariables, grid, timestepInformation);
        numberSaved++;
    }
}

}
}
