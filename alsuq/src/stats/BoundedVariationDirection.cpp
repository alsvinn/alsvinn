#include "alsuq/stats/BoundedVariationDirection.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq { namespace stats {



BoundedVariationDirection::BoundedVariationDirection(const alsuq::stats::StatisticsParameters &parameters)
    : StatisticsHelper(parameters)
{

}

std::vector<std::string> BoundedVariationDirection::getStatisticsNames() const
{
    return statisticsNames;
}

void BoundedVariationDirection::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                         const alsfvm::volume::Volume &extraVariables,
                                         const alsfvm::grid::Grid &grid,
                                         const alsfvm::simulator::TimestepInformation &timestepInformation)
{
    auto& bvX = findOrCreateSnapshot("bv_x",
                                      timestepInformation,
                                      conservedVariables,
                                      extraVariables,1,1,1, "cpu");

    auto& bvY = findOrCreateSnapshot("bv_y",
                                      timestepInformation,
                                      conservedVariables,
                                      extraVariables,1,1,1, "cpu");

    auto& bvZ = findOrCreateSnapshot("bv_z",
                                      timestepInformation,
                                      conservedVariables,
                                      extraVariables,1,1,1, "cpu");

    for (int var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        bvX.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(0);

        if (conservedVariables.getDimensions()>1) {
            bvY.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(1);
        }
        if (conservedVariables.getDimensions() > 2) {
             bvZ.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(2);
        }

    }

    for (int var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        bvX.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(0);

         if (conservedVariables.getDimensions()>1) {
              bvY.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(1);
         }

         if(conservedVariables.getDimensions() > 2) {
             bvZ.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation(2);
        }
    }







}

void BoundedVariationDirection::finalize()
{

}
REGISTER_STATISTICS(cpu, bv_direction, BoundedVariationDirection);
REGISTER_STATISTICS(cuda, bv_direction, BoundedVariationDirection);
}
}

