#include "alsuq/stats/StructureTwoPoints.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq {
namespace stats {

StructureTwoPoints::StructureTwoPoints(const StatisticsParameters& parameters)
    : StatisticsHelper(parameters),
      direction1(parameters.getInteger("direction1")),
      direction2(parameters.getInteger("direction2")),
      directionVector1(make_direction_vector(direction1)),
      directionVector2(make_direction_vector(direction2)),
      numberOfH(parameters.getInteger("numberOfH")),
      statisticsName ("structure_2pt")

{

}

std::vector<std::string> StructureTwoPoints::getStatisticsNames() const {
    return {statisticsName};
}

void StructureTwoPoints::computeStatistics(const alsfvm::volume::Volume&
    conservedVariables,
    const alsfvm::volume::Volume& extraVariables,
    const alsfvm::grid::Grid& grid,
    const alsfvm::simulator::TimestepInformation& timestepInformation) {
    auto& structure = this->findOrCreateSnapshot(statisticsName,
            timestepInformation,
            conservedVariables, extraVariables,
            numberOfH, numberOfH, 1);


    computeStructure(*structure.getVolumes().getConservedVolume(),
        conservedVariables);
    computeStructure(*structure.getVolumes().getExtraVolume(),
        extraVariables);
}

void StructureTwoPoints::finalizeStatistics() {

}

void StructureTwoPoints::computeStructure(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input) {
    for (size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        int ngx = input.getNumberOfXGhostCells();
        int ngy = input.getNumberOfYGhostCells();
        int ngz = input.getNumberOfZGhostCells();

        int nx = int(input.getNumberOfXCells()) - 2 * ngx;
        int ny = int(input.getNumberOfYCells()) - 2 * ngy;
        int nz = int(input.getNumberOfZCells()) - 2 * ngz;

        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    auto u_ijk = inputView.at(i + ngx, j + ngy, k + ngz);

                    for (int h1 = 0; h1 < int(numberOfH); ++h1) {
                        // For now we assume neumann boundary conditions
                        auto u_ijk_h1 = inputView.at(std::min(i + h1 * directionVector1.x,
                                    nx - 1) + ngx,
                                std::min(j + h1 * directionVector1.y, ny - 1) + ngy,
                                std::min(k + h1 * directionVector1.z, nz - 1) + ngz);

                        for (int h2 = 0; h2 < int(numberOfH); ++h2) {
                            auto u_ijk_h2 = inputView.at(std::min(i + h2 * directionVector1.x,
                                        nx - 1) % nx + ngx,
                                    std::min(j + h2 * directionVector1.y, ny - 1) % ny + ngy,
                                    std::min(k + h2 * directionVector1.z, nz - 1) % nz + ngz);


                            outputView.at(h1, h2, 0) += (u_ijk_h1 - u_ijk) * (u_ijk_h1 - u_ijk) *
                                (u_ijk_h2 - u_ijk) / (nx * ny * nz);
                        }
                    }
                }
            }
        }


    }
}
REGISTER_STATISTICS(cpu, structure_2pt, StructureTwoPoints)
}
}
