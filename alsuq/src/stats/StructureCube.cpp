#include "alsuq/stats/StructureCube.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq { namespace stats {

StructureCube::StructureCube(const StatisticsParameters &parameters)
    : StatisticsHelper(parameters),
      p(parameters.getParameterAsDouble("p")),
      numberOfH(parameters.getParameterAsInteger("numberOfH")),
      statisticsName ("structure_basic_" + std::to_string(p))

{

}

std::vector<std::string> StructureCube::getStatisticsNames() const
{
    return {statisticsName};
}

void StructureCube::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
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

void StructureCube::finalize()
{

}

void StructureCube::computeStructure(alsfvm::volume::Volume &output,
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
                    for(int h = 1; h < numberOfH; ++h) {

                        computeCube(outputView, inputView, i,j,k,h, nx, ny, nz,
                                    ngx, ngy, ngz, input.getDimensions());

                    }
                }
            }
        }


    }
}

void StructureCube::computeCube(alsfvm::memory::View<real> &output,
                                const alsfvm::memory::View<const real> &input,
                                int i, int j, int k, int h, int nx, int ny, int nz,
                                int ngx, int ngy, int ngz, int dimensions)
{



    auto makePositive = [](int position, int N) {
        if (position < 0) {
            position += N;
        }
        return position;
    };

    const auto u = input.at(i + ngx,j + ngy,k + ngz);
    for (int d = 0; d < dimensions; d++) {
        // side = 0 represents bottom, side = 1 represents top
        for (int side = 0; side < 2; side++) {
            const bool zDir = (d == 2);
            const bool yDir = (d == 1);
            const bool xDir = (d == 0);
            // Either we start on the left (i == 0), or on the right(i==1)
            const int zStart = zDir ?
                (side == 0 ? k-h : k+h) : (dimensions > 2 ? k-h : 0);

            const int zEnd = zDir ?
                (zStart + 1) : (dimensions > 2 ? k+h+1 : 1);

            const int yStart = yDir ?
                (side == 0 ? j - h : j + h + 1) : (dimensions > 1 ? j - h : 0);

            const int yEnd = yDir ?
                (yStart + 1) : (dimensions > 1 ? j+h+1 : 1);

            const int xStart = xDir ?
                (side == 0 ? i - h : i + h + 1) : i - h;

            const int xEnd = xDir ?
                (xStart + 1) : i + h + 1;

            for (int z = zStart; z < zEnd; z++) {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = xStart; x < xEnd; x++) {
                        const auto u_h = input.at(makePositive(x, nx)%nx + ngx,
                                                  makePositive(y, ny)%ny + ngy,
                                                  makePositive(z, nz)%nz + ngz);
                        output.at(h) += std::pow(std::abs(u_h-u),p)/(nx*ny*nz);
                    }
                }
            }
        }
    }
}
REGISTER_STATISTICS(cpu, structure_cube, StructureCube)
}
}
