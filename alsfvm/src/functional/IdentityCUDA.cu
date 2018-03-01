#include "alsfvm/functional/IdentityCUDA.hpp"
#include "alsfvm/functional/register_functional.hpp"

namespace alsfvm {
namespace functional {

namespace {

__global__ void addInnerKernel(memory::View<real> output,
                               memory::View<const real> input,
                               int ngx, int ngy, int ngz,
                               double weight)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( index >= output.size()) {
        return;
    }

    const size_t x = index % output.getNumberOfXCells();
    const size_t y = (index / output.getNumberOfXCells()) % output.getNumberOfYCells();
    const size_t z = index / (output.getNumberOfXCells()*output.getNumberOfYCells());
    output.at(index) += weight*input.at(x + ngx, y + ngy, z + ngz);
}

}
IdentityCUDA::IdentityCUDA(const Functional::Parameters& parameters) {

}

void IdentityCUDA::operator()(volume::Volume& conservedVolumeOut,
    volume::Volume& extraVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const volume::Volume& extraVolumeIn,
    const real weight,
    const grid::Grid& ) {

    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();
    for (size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {


        auto viewIn = conservedVolumeIn.getScalarMemoryArea(var)->getView();
        auto viewOut = conservedVolumeOut.getScalarMemoryArea(var)->getView();

        const size_t threads = 1024;
        const size_t size = viewOut.size();
        addInnerKernel<<<(size + threads -1)/threads, threads>>>(viewOut, viewIn,
                                                                ghostCells.x,
                                                                ghostCells.y,
                                                                ghostCells.z,
                                                                weight);

    }

    for (size_t var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {



        auto viewIn = extraVolumeIn.getScalarMemoryArea(var)->getView();
        auto viewOut = extraVolumeOut.getScalarMemoryArea(var)->getView();

        const size_t threads = 1024;
        const size_t size = viewOut.size();
        addInnerKernel<<<(size + threads -1)/threads, threads>>>(viewOut, viewIn,
                                                                ghostCells.x,
                                                                ghostCells.y,
                                                                ghostCells.z,
                                                                weight);

    }
}

ivec3 IdentityCUDA::getFunctionalSize(const grid::Grid& grid) const {
    return grid.getDimensions();
}
REGISTER_FUNCTIONAL(cuda, identity, IdentityCUDA)
}
}
