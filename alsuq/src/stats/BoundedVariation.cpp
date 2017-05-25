#include "alsuq/stats/BoundedVariation.hpp"

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
                                      extraVariables);


    

}

void BoundedVariation::finalize()
{

}

}
}
