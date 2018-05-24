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
namespace  mpi {
namespace cartesian {

//! Computes the rank from the 3d coordinates (this is for domain decomposition
//! to know where your CPU is "in space")
//!
//! @param coordinate the coordinates
//! @param numberOfProcessors the number of processors in each direction
//!
//! @see getCoordinates
int getRankIndex(const ivec3& coordinate, const ivec3& numberOfProcessors) {
    int x = coordinate.x;
    int y = coordinate.y;
    int z = coordinate.z;

    const int nx = numberOfProcessors.x;
    const int ny = numberOfProcessors.y;
    const int nz = numberOfProcessors.z;

    if (x < 0) {
        x += nx;
    }

    if (x > nx - 1) {
        x -= nx;
    }

    if (y < 0) {
        y += ny;
    }

    if (y > ny - 1) {
        y -= ny;
    }


    if (z < 0) {
        z += nz;
    }

    if (z > nz - 1) {
        z -= nz;
    }

    return x + y * nx + z * nx * ny;
}
}
}
}
