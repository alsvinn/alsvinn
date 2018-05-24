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

#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace mpi {
namespace cartesian {
//! Returns the number of segment needed for the MPI_Type
//!
//! @param side an integer from 0 to 5 (inclusive)
//! Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
//! -------|------------------|-----------------|-----------------
//!    0   |       left       |     left        |    left
//!    1   |       right      |     right       |    right
//!    2   |     < not used > |     bottom      |    bottom
//!    3   |     < not used > |     top         |    top
//!    4   |     < not used > |   < not used >  |    front
//!    5   |     < not used > |   < not used >  |    back
//!
//! @param dimensions the number of dimension (from 1 to 3)
//!
//! @param numberOfCellsPerDirection for each direction, list the total number of cells in that direction
//!                                  (this includes ghost cells)
//!
inline int computeNumberOfSegments(int side, int dimensions,
    ivec3 numberOfCellsPerDirection) {
    int numberOfSegments = 1;


    if ( side < 2) { // x side
        if (dimensions == 1) {
            numberOfSegments = 1;
        } else if (dimensions == 2) {
            numberOfSegments = numberOfCellsPerDirection.y;
        } else {
            numberOfSegments = numberOfCellsPerDirection.y * numberOfCellsPerDirection.z;
        }
    } else if (side < 4) { // y side
        if (dimensions == 2) {
            numberOfSegments = 1;
        } else {
            numberOfSegments = numberOfCellsPerDirection.z;
        }

    } else { // z side
        numberOfSegments = 1;
    }

    return numberOfSegments;
}

}
}
}
