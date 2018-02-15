#include "alsuq/stats/BoundedVariation.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {



BoundedVariation::BoundedVariation(const alsuq::stats::StatisticsParameters&
    parameters)
    : StatisticsHelper(parameters), p(parameters.getParameterAsInteger("p")),
      statisticsName("bv_" + std::to_string(p)) {

}

std::vector<std::string> BoundedVariation::getStatisticsNames() const {
    return {statisticsName};
}

void BoundedVariation::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& bv = findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    const ivec3 start = conservedVariables.getNumberOfGhostCells();
    const ivec3 end = conservedVariables.getSize() -
        conservedVariables.getNumberOfGhostCells();

    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        bv.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0]
            = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(p,
                    start,
                    end);
    }

    for (size_t var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        bv.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0]
            = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(p,
                    start,
                    end);
    }






}

void BoundedVariation::finalizeStatistics() {

}
REGISTER_STATISTICS(cpu, bv, BoundedVariation);
REGISTER_STATISTICS(cuda, bv, BoundedVariation);
}
}
