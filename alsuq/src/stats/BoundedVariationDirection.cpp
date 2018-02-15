#include "alsuq/stats/BoundedVariationDirection.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {



BoundedVariationDirection::BoundedVariationDirection(const
    alsuq::stats::StatisticsParameters& parameters)
    : StatisticsHelper(parameters), p(parameters.getParameterAsInteger("p")),
      statisticsNames({"bv_x_" + std::to_string(p),
    "bv_y_" + std::to_string(p),
    "bv_z_" + std::to_string(p)}) {

}

std::vector<std::string> BoundedVariationDirection::getStatisticsNames() const {
    return statisticsNames;
}

void BoundedVariationDirection::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    const ivec3 start = conservedVariables.getNumberOfGhostCells();
    const ivec3 end = conservedVariables.getSize() -
        conservedVariables.getNumberOfGhostCells();

    auto& bvX = findOrCreateSnapshot(statisticsNames[0],
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    auto& bvY = findOrCreateSnapshot(statisticsNames[1],
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    auto& bvZ = findOrCreateSnapshot(statisticsNames[2],
            timestepInformation,
            conservedVariables,
            extraVariables, 1, 1, 1, "cpu");

    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        bvX.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0]
            = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(0, p, start,
                    end);

        if (conservedVariables.getDimensions() > 1) {
            bvY.getVolumes().getConservedVolume()->getScalarMemoryArea(
                var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(
                        var)->getTotalVariation(1, p, start, end);
        }

        if (conservedVariables.getDimensions() > 2) {
            bvZ.getVolumes().getConservedVolume()->getScalarMemoryArea(
                var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(
                        var)->getTotalVariation(2, p, start, end);
        }

    }

    for (size_t var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        bvX.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] =
            conservedVariables.getScalarMemoryArea(var)->getTotalVariation(0, p, start,
                end);

        if (conservedVariables.getDimensions() > 1) {
            bvY.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] =
                conservedVariables.getScalarMemoryArea(var)->getTotalVariation(1, p, start,
                    end);
        }

        if (conservedVariables.getDimensions() > 2) {
            bvZ.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] =
                conservedVariables.getScalarMemoryArea(var)->getTotalVariation(2, p, start,
                    end);
        }
    }







}

void BoundedVariationDirection::finalizeStatistics() {

}
REGISTER_STATISTICS(cpu, bv_direction, BoundedVariationDirection);
REGISTER_STATISTICS(cuda, bv_direction, BoundedVariationDirection);
}
}

