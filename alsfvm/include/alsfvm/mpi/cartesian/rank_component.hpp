#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm { namespace  mpi {
namespace cartesian {

//! Computes the 3d coordinates of a linear rank (this is for domain decomposition
//! to know where your CPU is "in space")
//!
//! @param rank the mpi rank
//! @param numberOfProcessors the number of processors in each direction
//!
//! @see getRankIndex
ivec3 getCoordinates(int rank, const ivec3& numberOfProcessors) {
    return ivec3{rank%numberOfProcessors.x,
    (rank/numberOfProcessors.x)%numberOfProcessors.y,
    rank/(numberOfProcessors.x*numberOfProcessors.y)};
}
}
}
}
