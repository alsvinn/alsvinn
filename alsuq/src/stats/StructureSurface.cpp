#include "alsuq/stats/StructureSurface.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {

StructureSurface::StructureSurface(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      p(parameters.getDouble("p")),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_surface_" + std::to_string(p))

{

}

std::vector<std::string> StructureSurface::getStatisticsNames() const {
    return {statisticsName};
}

void StructureSurface::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables, extraVariables,
            numberOfH, 1, 1);


    computeStructure(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
    computeStructure(*structure.getVolumes().getExtraVolume(),
        extraVariables);
}

void StructureSurface::finalizeStatistics() {

}

void StructureSurface::computeStructure(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input) {
    THROW("Not implemented yet");
}
REGISTER_STATISTICS(cpu, structure_surface, StructureSurface)
}
}
