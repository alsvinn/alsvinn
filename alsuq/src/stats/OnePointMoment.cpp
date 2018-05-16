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

