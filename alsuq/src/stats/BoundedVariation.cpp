#include "alsuq/stats/BoundedVariation.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq { namespace stats {



BoundedVariation::BoundedVariation(const alsuq::stats::StatisticsParameters &parameters)
    : StatisticsHelper(parameters)
{

}

std::vector<std::string> BoundedVariation::getStatisticsNames() const
{
    return {statisticsName};
}

void BoundedVariation::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                         const alsfvm::volume::Volume &extraVariables,
                                         const alsfvm::grid::Grid &grid,
                                         const alsfvm::simulator::TimestepInformation &timestepInformation)
{
    auto& bv = findOrCreateSnapshot("bv",
                                      timestepInformation,
                                      conservedVariables,
                                      extraVariables,1,1,1, "cpu");

    for (int var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {
        bv.getVolumes().getConservedVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation();
    }

    for (int var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        bv.getVolumes().getExtraVolume()->getScalarMemoryArea(var)->getPointer()[0] = conservedVariables.getScalarMemoryArea(var)->getTotalVariation();
    }




    

}

void BoundedVariation::finalize()
{

}
REGISTER_STATISTICS(cpu, bv, BoundedVariation);
REGISTER_STATISTICS(cuda, bv, BoundedVariation);
}
}
