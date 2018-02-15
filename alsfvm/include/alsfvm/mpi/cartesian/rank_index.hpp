
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
