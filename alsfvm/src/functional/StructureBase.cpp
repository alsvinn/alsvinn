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

#include "alsfvm/functional/StructureBase.hpp"
#include "alsfvm/functional/structure_common.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include "alsutils/log.hpp"

namespace alsfvm {
namespace functional {

StructureBase::StructureBase(const Functional::Parameters& parameters)
    : direction(parameters.getInteger("direction")),
      p(parameters.getDouble("p")),
      directionVector(make_direction_vector(direction)),
      numberOfH(parameters.getInteger("numberOfH")) {

}

void StructureBase::operator()(volume::Volume& conservedVolumeOut,
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

ivec3 StructureBase::getFunctionalSize(const grid::Grid& grid) const {
    ALSVINN_LOG(INFO, "numberOfH = " << numberOfH);
    return {int(numberOfH), 1, 1};
}

void StructureBase::computeStructure(volume::Volume& output,
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
                    for (int h = 0; h < int(numberOfH); ++h) {


                        auto u_ijk = inputView.at(i + ngx, j + ngy, k + ngz);





                        // For now we assume periodic boundary conditions
                        auto u_ijk_h = inputView.at((i + h * directionVector.x) % nx + ngx,
                                (j + h * directionVector.y) % ny + ngy,
                                (k + h * directionVector.z) % nz + ngz);



                        outputView.at(h, 0, 0) += std::pow(std::abs(u_ijk - u_ijk_h),
                                p) / (nx * ny * nz);
                    }
                }
            }
        }


    }
}



REGISTER_FUNCTIONAL(cpu, structure_base, StructureBase)
}
}
