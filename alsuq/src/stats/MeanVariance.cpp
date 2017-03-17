#include "alsuq/stats/MeanVariance.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq { namespace stats {

MeanVariance::MeanVariance(const StatisticsParameters &parameters)
{

}

std::vector<std::string> MeanVariance::getStatisticsNames() const
{
    return  {"mean", "variance"};
}

void MeanVariance::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                     const alsfvm::volume::Volume &extraVariables,
                                     const alsfvm::grid::Grid &grid,
                                     const alsfvm::simulator::TimestepInformation &timestepInformation)
{

    auto& mean = findOrCreateSnapshot("mean",
                                      timestepInformation,
                                      conservedVariables,
                                      extraVariables);

    auto& variance = findOrCreateSnapshot("variance",
                                          timestepInformation,
                                          conservedVariables,
                                          extraVariables);



    *mean.getVolumes().getConservedVolume() += conservedVariables;
    *mean.getVolumes().getExtraVolume() += extraVariables;

    variance.getVolumes().getConservedVolume()->addPower(conservedVariables, 2);
    variance.getVolumes().getExtraVolume()->addPower(extraVariables, 2);


}

REGISTER_STATISTICS(cpu, meanvar, MeanVariance);
REGISTER_STATISTICS(cuda, meanvar, MeanVariance);
}
}

