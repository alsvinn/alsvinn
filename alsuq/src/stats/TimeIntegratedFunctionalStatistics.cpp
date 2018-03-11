#include "alsuq/stats/TimeIntegratedFunctionalStatistics.hpp"
#include "alsfvm/functional/FunctionalFactory.hpp"
#include "alsuq/stats/stats_util.hpp"

namespace alsuq {
namespace stats {

TimeIntegratedFunctionalStatistics::TimeIntegratedFunctionalStatistics(
    const StatisticsParameters& parameters)
    : StatisticsHelper(parameters) {

    alsfvm::functional::FunctionalFactory functionalFactory;

    platform = parameters.getPlatform();
    const std::string name = parameters.getString("functionalName");
    functional = functionalFactory.makeFunctional(platform, name,
            parameters);

    statisticsNames = {"mean_" + name};

    time = parameters.getDouble("time");
    timeRadius = parameters.getDouble("timeRadius");

    fixedTimestepInformation = alsfvm::simulator::TimestepInformation(time, 0);

}

std::vector<std::string>
TimeIntegratedFunctionalStatistics::getStatisticsNames()
const {
    return statisticsNames;
}

void TimeIntegratedFunctionalStatistics::computeStatistics(
    const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {


    const double currentTime = timestepInformation.getCurrentTime();

    auto gridFunctional = functional->getFunctionalSize(grid);
    auto& functionalTime = this->findOrCreateSnapshot(statisticsNames[0],
            fixedTimestepInformation,
            conservedVariables, extraVariables,
            gridFunctional.x, gridFunctional.y, gridFunctional.z, platform);

    if (std::abs(currentTime - time) <= timeRadius) {
        // Now we should write
        double dt = currentTime - lastTime;

        // If dt <= 0, we get no contributioni (usually it is because we
        // are on a new sample)
        if (dt > 0) {
            functional->operator ()(*functionalTime.getVolumes().getConservedVolume(),
                *functionalTime.getVolumes().getExtraVolume(),
                conservedVariables,
                extraVariables,
                dt,
                grid
            );
        }

    }


}

void TimeIntegratedFunctionalStatistics::finalizeStatistics() {

}

REGISTER_STATISTICS(cuda, functional_time_integrated,
    TimeIntegratedFunctionalStatistics)
REGISTER_STATISTICS(cpu, functional_time_integrated,
    TimeIntegratedFunctionalStatistics)


}
}
