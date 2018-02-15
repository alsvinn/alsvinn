#include "alsuq/stats/StatisticsTimer.hpp"
#include "alsutils/log.hpp"
#include <chrono>

namespace alsuq {
namespace stats {

StatisticsTimer::StatisticsTimer(const std::string& name,
    std::shared_ptr<Statistics> statistics)
    : name(name), statistics(statistics) {

}

StatisticsTimer::~StatisticsTimer() {
    ALSVINN_LOG(INFO, "Statistics times for " << name << ":\n"
        << "\tcomputing:  " << statisticsTime << " s\n"
        << "\tcombing:    " << combineTime << " s\n"
        << "\tfinalizing: " << finalizeTime << " s\n");
}

void StatisticsTimer::combineStatistics() {
    auto startTime = std::chrono::high_resolution_clock::now();
    statistics->combineStatistics();
    auto endTime = std::chrono::high_resolution_clock::now();

    combineTime += std::chrono::duration_cast<std::chrono::seconds>
        (endTime - startTime).count();

}

void StatisticsTimer::addWriter(const std::string& name,
    std::shared_ptr<alsfvm::io::Writer>& writer) {
    statistics->addWriter(name, writer);
}

std::vector<std::string> StatisticsTimer::getStatisticsNames() const {
    return statistics->getStatisticsNames();
}

void StatisticsTimer::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {

    auto startTime = std::chrono::high_resolution_clock::now();
    statistics->computeStatistics(conservedVariables, extraVariables, grid,
        timestepInformation);
    auto endTime = std::chrono::high_resolution_clock::now();

    statisticsTime += std::chrono::duration_cast<std::chrono::seconds>
        (endTime - startTime).count();

}

void StatisticsTimer::finalizeStatistics() {
    auto startTime = std::chrono::high_resolution_clock::now();
    statistics->finalizeStatistics();
    auto endTime = std::chrono::high_resolution_clock::now();

    finalizeTime += std::chrono::duration_cast<std::chrono::seconds>
        (endTime - startTime).count();
}

void StatisticsTimer::writeStatistics(const alsfvm::grid::Grid& grids) {
    statistics->writeStatistics(grids);
}

}
}
