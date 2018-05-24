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

//! Computes the 3d coordinates of a linear rank (this is for domain decomposition
//! to know where your CPU is "in space")
//!
//! @param rank the mpi rank
//! @param numberOfProcessors the number of processors in each direction
//!
//! @see getRankIndex
ivec3 getCoordinates(int rank, const ivec3& numberOfProcessors) {
    return ivec3{rank % numberOfProcessors.x,
            (rank / numberOfProcessors.x) % numberOfProcessors.y,
            rank / (numberOfProcessors.x * numberOfProcessors.y)};
}
}
}
}
