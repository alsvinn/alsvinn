#include "alsuq/stats/FunctionalStatistics.hpp"
#include "alsfvm/functional/FunctionalFactory.hpp"
#include "alsuq/stats/stats_util.hpp"

namespace alsuq {
namespace stats {

FunctionalStatistics::FunctionalStatistics(
    const StatisticsParameters& parameters)
    : StatisticsHelper(parameters) {

    alsfvm::functional::FunctionalFactory functionalFactory;

    platform = parameters.getPlatform();
    const std::string name = parameters.getString("functionalName");
    functional = functionalFactory.makeFunctional(platform, name,
            parameters);

    statisticsNames = {"mean_" + name};

}

std::vector<std::string> FunctionalStatistics::getStatisticsNames()
const {
    return statisticsNames;
}

void FunctionalStatistics::computeStatistics(
    const alsfvm::volume::Volume& conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {



    auto gridFunctional = functional->getFunctionalSize(grid);
    auto& functionalData = this->findOrCreateSnapshot(statisticsNames[0],
            timestepInformation,
            conservedVariables, extraVariables,
            gridFunctional.x, gridFunctional.y, gridFunctional.z, platform);





    functional->operator ()(*functionalData.getVolumes().getConservedVolume(),
        *functionalData.getVolumes().getExtraVolume(),
        conservedVariables,
        extraVariables,
        1,
        grid
    );


}


void FunctionalStatistics::finalizeStatistics() {

}

REGISTER_STATISTICS(cuda, functional,
    FunctionalStatistics)
REGISTER_STATISTICS(cpu, functional,
    FunctionalStatistics)


}
}
