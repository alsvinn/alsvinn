#pragma once

#include <vector>
#include "alsfvm/types.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"

namespace alsfvm {
namespace mpi {
namespace cartesian {
//! Computes the lengths (in number of reals) of the segments
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
//! @param ghostCells number of ghost cells in the given direction
inline std::vector<int> computeLengths(int side, int dimensions,
    ivec3 numberOfCellsPerDirection,
    int ghostCells) {

    const int numberOfSegments = computeNumberOfSegments(side, dimensions,
            numberOfCellsPerDirection);
    std::vector<int> lengths(numberOfSegments, 0);

    for (int i = 0; i < numberOfSegments; ++i) {
        if (dimensions == 1) {
            lengths[i] = ghostCells;

        } else if (dimensions == 2) {
            if ( side < 2) {


                lengths[i] = ghostCells;
            } else {

                lengths[i] = ghostCells * numberOfCellsPerDirection.x;
            }
        } else {
            if ( side < 2) {

                lengths[i] = ghostCells;
            } else if (side < 4) {

                lengths[i] = ghostCells * numberOfCellsPerDirection.x;
            } else {


                lengths[i] = ghostCells * numberOfCellsPerDirection.x *
                    numberOfCellsPerDirection.y;
            }
        }
    }

    return lengths;
}
}
}
}
