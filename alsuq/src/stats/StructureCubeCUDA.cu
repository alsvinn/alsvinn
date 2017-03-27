#include "alsuq/stats/StructureCubeCUDA.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsuq/stats/stats_util.hpp"
namespace alsuq { namespace stats {

namespace {

__device__ int makePositive(int index, int N) {
    if (index < 0) {
        return index += N;
    }
    return index;
}

//! Computes the structure function for FIXED h
//!
//! The goal is to compute the structure function, then reduce (sum) over space
//! then go on to next h
__global__ void computeStructureCube(real* output,
                                     alsfvm::memory::View<const real> input,
                                     int h,
                                     int nx, int ny, int nz, int ngx, int ngy, int ngz,
                                     real p, int dimensions)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nx*ny*nz) {
        return;
    }

    const int i = index % nx;
    const int j = (index / nx) % ny;
    const int k = (index / nx / ny);


    const auto u = input.at(i + ngx, j + ngy, k + ngz);
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
                        output[index] += pow(fabs(u_h-u),p);
                    }
                }
            }
        }
    }
}
}

StructureCubeCUDA::StructureCubeCUDA(const StatisticsParameters &parameters)
    : StatisticsHelper(parameters),
      p(parameters.getParameterAsDouble("p")),
      numberOfH(parameters.getParameterAsInteger("numberOfH")),
      statisticsName ("structure_basic_" + std::to_string(p))

{

}

std::vector<std::string> StructureCubeCUDA::getStatisticsNames() const
{
    return {statisticsName};
}

void StructureCubeCUDA::computeStatistics(const alsfvm::volume::Volume &conservedVariables,
                                          const alsfvm::volume::Volume &extraVariables,
                                          const alsfvm::grid::Grid &grid,
                                          const alsfvm::simulator::TimestepInformation &timestepInformation)
{
    auto& structure = this->findOrCreateSnapshot(statisticsName, timestepInformation,
                                                 conservedVariables, extraVariables,
                                                 numberOfH, 1, 1, "cpu");


    computeStructure(*structure.getVolumes().getConservedVolume(),
                     conservedVariables);
    computeStructure(*structure.getVolumes().getExtraVolume(),
                     extraVariables);
}

void StructureCubeCUDA::finalize()
{

}

void StructureCubeCUDA::computeStructure(alsfvm::volume::Volume &output,
                                         const alsfvm::volume::Volume &input)
{
    for(size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        const int ngx = input.getNumberOfXGhostCells();
        const int ngy = input.getNumberOfYGhostCells();
        const int ngz = input.getNumberOfZGhostCells();

        const int nx = int(input.getNumberOfXCells()) - 2 * ngx;
        const int ny = int(input.getNumberOfYCells()) - 2 * ngy;
        const int nz = int(input.getNumberOfZCells()) - 2 * ngz;

        structureOutput.resize(nx*ny*nz,0);
        const int dimensions = input.getDimensions();
        for(int h = 1; h < numberOfH; ++h) {
            const int threads = 1024;
            const int size = nx*ny*nz;
            const int blockNumber = (size+threads -1)/threads;

            computeStructureCube<<<blockNumber,threads>>>(thrust::raw_pointer_cast(structureOutput.data()),
                                                          inputView,
                                                          h, nx, ny, nz, ngx, ngy, ngz, p, dimensions);

            real structureResult = thrust::reduce(structureOutput.begin(),
                                                  structureOutput.end(),
                                                  0.0, thrust::plus<real>());

            outputView.at(h) += structureResult/(nx*ny*nz);
        }

    }
}

REGISTER_STATISTICS(cuda, structure_cube, StructureCubeCUDA)
}
}
