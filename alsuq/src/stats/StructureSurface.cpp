#include "alsuq/stats/StructureSurface.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq { namespace stats {

StructureSurface::StructureSurface(const StatisticsParameters &parameters)
    : StatisticsHelper(parameters),
      p(parameters.getParameterAsDouble("p")),
      numberOfH(parameters.getParameterAsInteger("numberOfH")),
      statisticsName ("structure_surface_" + std::to_string(p))

{

}

std::vector<std::string> StructureSurface::getStatisticsNames() const
{
    return {statisticsName};
}

void StructureSurface::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                       const alsfvm::volume::Volume &extraVariables,
                                       const alsfvm::grid::Grid &grid,
                                       const alsfvm::simulator::TimestepInformation &timestepInformation)
{
    auto& structure = this->findOrCreateSnapshot(statisticsName, timestepInformation,
                                                 conservedVariables, extraVariables,
                                                 numberOfH, 1, 1);


    computeStructure(*structure.getVolumes().getConservedVolume(),
                     conservedVariables);
    computeStructure(*structure.getVolumes().getExtraVolume(),
                     extraVariables);
}

void StructureSurface::finalize()
{

}

void StructureSurface::computeStructure(alsfvm::volume::Volume &output,
                                      const alsfvm::volume::Volume &input)
{
    for(size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        int ngx = input.getNumberOfXGhostCells();
        int ngy = input.getNumberOfYGhostCells();
        int ngz = input.getNumberOfZGhostCells();

        int nx = int(input.getNumberOfXCells()) - 2 * ngx;
        int ny = int(input.getNumberOfYCells()) - 2 * ngy;
        int nz = int(input.getNumberOfZCells()) - 2 * ngz;
        for(int k = 0; k < nz; ++k) {
            for(int j = 0; j < ny; ++j) {
                for(int i = 0; i < nx; ++i) {
                    for(int h = 0; h < numberOfH; ++h) {


                        auto u_ijk = inputView.at(i+ngx,j+ngy,k+ngz);

                        // For now we assume periodic boundary conditions
                        auto u_ijk_h = inputView.at((i + h*directionVector.x)%nx + ngx,
                                                    (j + h*directionVector.y)%ny + ngy,
                                                    (k + h*directionVector.z)%nz + ngz);



                        outputView.at(h,0,0) += std::pow(std::abs(u_ijk-u_ijk_h),p)/(nx*ny*nz);
                    }
                }
            }
        }


    }
}
REGISTER_STATISTICS(cpu, structure_surface, StructureSurface)
}
}
