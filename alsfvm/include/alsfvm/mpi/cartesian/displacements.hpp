#pragma once

#include <vector>
#include "alsfvm/types.hpp"
#include "alsfvm/mpi/cartesian/number_of_segments.hpp"

namespace alsfvm {
namespace mpi {
namespace cartesian {
//! Computes the displacmenets (offsets) of the segments
//!
//! @param side an integer from 0 to 5 (inclusive)
//!  //! Index  |  Spatial side 1D | Spatial side 2D | Spatial side 3D
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
//!
//! @param baseOffset the offset to use in the direction. This is to handle the receive and send
//!                   types. This should always be of non-negative quantity, the sign will be deduced
inline std::vector<int> computeDisplacements(int side, int dimensions, ivec3 numberOfCellsPerDirection,
                                      int ghostCells, int baseOffset) {

    const int numberOfSegments = computeNumberOfSegments(side, dimensions,
                                                         numberOfCellsPerDirection);

    if (side % 2 == 1) { // we should subtract the offset since we are on a "right side/top side/back side"
        baseOffset *= -1;
    }

    std::vector<int> displacements(numberOfSegments, 0);
    for (int i = 0; i < numberOfSegments; ++i) {
        if (dimensions == 1) {
            displacements[i] = side * (numberOfCellsPerDirection.x - ghostCells)
                                       +baseOffset;

        } else if(dimensions == 2) {
            if ( side < 2) {

                if (side > 0 || i > 0) {
                    displacements[i] = (numberOfCellsPerDirection.x);
                }
                if (i == 0 && side == 1) {
                    displacements[i] -= ghostCells;
                }
                displacements[i] += baseOffset;
                if (i > 0) {
                    displacements[i] += displacements[i-1];
                }

            } else {
                displacements[i] += baseOffset * numberOfCellsPerDirection.x;

                if (side > 2) { // bottom side does not have any displacement
                    // we only have two segments in dimension 2 for the y-direction,
                    // and the first one is 0 displacement, therefore, we do not add
                    // displacements[i-1]
                    displacements[i] += numberOfCellsPerDirection.x*(numberOfCellsPerDirection.y+baseOffset)
                            - numberOfCellsPerDirection.x*ghostCells;
                }

            }
        } else {
            if ( side < 2) {
                displacements[i] += baseOffset;
                if (side > 0 || i > 0) {
                    displacements[i] += (numberOfCellsPerDirection.x);
                }
                if (i == 0 && side == 1) {
                    displacements[i] -= ghostCells;
                }
                if (i > 0) {
                    displacements[i] += displacements[i-1];
                }

            } else if (side < 4) {
                displacements[i] += baseOffset * numberOfCellsPerDirection.x;

                if (side > 2 || i > 0) {
                    displacements[i] = numberOfCellsPerDirection.x*(numberOfCellsPerDirection.y);
                }
                if (i == 0 && side == 3) {
                    displacements[i] -= ghostCells * numberOfCellsPerDirection.x;
                }

                if (i > 0) {
                    displacements[i] += displacements[i-1];
                }

            } else {

                // There is only one segment in the z direction, and it only needs
                // displacement if it is the back side
                if (side == 5) {
                    displacements[i] = numberOfCellsPerDirection.x*numberOfCellsPerDirection.y*(numberOfCellsPerDirection.z+baseOffset)-
                            numberOfCellsPerDirection.x*numberOfCellsPerDirection.y*ghostCells;
                }

            }
        }
    }
    return displacements;
}
}
}
}
