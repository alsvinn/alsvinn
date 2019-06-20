/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    const volume::Volume& conservedVolumeIn,
    const real weight,
    const grid::Grid& grid) {
    computeStructure(conservedVolumeOut,
        conservedVolumeIn);
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
