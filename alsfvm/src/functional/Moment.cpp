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

#include "alsfvm/functional/Moment.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <boost/math/special_functions/legendre.hpp>


namespace alsfvm {
namespace functional {

Moment::Moment(const Functional::Parameters& parameters)
    :
    degree(parameters.getInteger("degree")) {

}

void Moment::operator()(volume::Volume& conservedVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const real weight,
    const grid::Grid& grid) {



    const auto lengths = grid.getCellLengths();

    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();

    const auto innerSize = conservedVolumeIn.getInnerSize();

    conservedVolumeOut.addPower(conservedVolumeIn, degree, weight);
}

ivec3 Moment::getFunctionalSize(const grid::Grid& grid) const {
    return grid.getDimensions();
}

ivec3 Moment::getGhostCellSizes(const grid::Grid& grid,
    const volume::Volume& volume) const {
    return volume.getNumberOfGhostCells();
}

REGISTER_FUNCTIONAL(cpu, moment, Moment)
REGISTER_FUNCTIONAL(cuda, moment, Moment)
}
}
