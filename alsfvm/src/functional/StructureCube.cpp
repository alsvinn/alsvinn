#include "alsfvm/functional/StructureCube.hpp"
#include "alsfvm/functional/structure_common.hpp"
#include "alsfvm/functional/register_functional.hpp"

namespace alsfvm {
namespace functional {

StructureCube::StructureCube(const Functional::Parameters& parameters)
    :  p(parameters.getDouble("p")),
       numberOfH(parameters.getInteger("numberOfH"))

{

}

void StructureCube::operator()(volume::Volume& conservedVolumeOut,
    volume::Volume& extraVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const volume::Volume& extraVolumeIn,
    const real weight,
    const grid::Grid& grid) {
    computeStructure(conservedVolumeOut,
        conservedVolumeIn);
    computeStructure(extraVolumeOut,
        extraVolumeIn);
}

ivec3 StructureCube::getFunctionalSize(const grid::Grid& grid) const {
    return {numberOfH, 1, 1};
}

void StructureCube::computeStructure(volume::Volume& output,
    const volume::Volume& input) {
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
                    for (int h = 1; h < numberOfH; ++h) {

                        computeCube(outputView, inputView, i, j, k, h, nx, ny, nz,
                            ngx, ngy, ngz, input.getDimensions());

                    }
                }
            }
        }


    }
}

void StructureCube::computeCube(alsfvm::memory::View<real>& output,
    const alsfvm::memory::View<const real>& input,
    int i, int j, int k, int h,
    int nx, int ny, int nz,
    int ngx, int ngy, int ngz,
    int dimensions) {
    computeStructureCube(output, input, i, j, k, h, nx, ny, nz, ngx, ngy, ngz,
        dimensions, p);
}

REGISTER_FUNCTIONAL(cpu, structure_cube, StructureCube)
}
}
