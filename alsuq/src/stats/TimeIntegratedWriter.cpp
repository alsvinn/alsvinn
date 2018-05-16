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
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {




    statistics->computeStatistics(conservedVariables, extraVariables, grid,
        timestepInformation);

}

}
}
