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

#include "alsfvm/functional/StructureCubeCUDA.hpp"
#include "alsfvm/functional/structure_common_cuda.hpp"
#include "alsfvm/functional/register_functional.hpp"

namespace alsfvm {
namespace functional {

StructureCubeCUDA::StructureCubeCUDA(const Functional::Parameters& parameters)
    :  p(parameters.getDouble("p")),
       numberOfH(parameters.getInteger("numberOfH"))

{

}

void StructureCubeCUDA::operator()(volume::Volume& conservedVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const real weight,
    const grid::Grid& grid) {

    conservedVolumeOut.makeZero();

    alsfvm::functional::dispatchComputeStructureCubeCUDA(conservedVolumeOut,
                                              conservedVolumeIn, structureOutput, numberOfH, p);


}

ivec3 StructureCubeCUDA::getFunctionalSize(const grid::Grid& grid) const {
    return {numberOfH, 1, 1};
}


REGISTER_FUNCTIONAL(cuda, structure_cube, StructureCubeCUDA)
}
}
