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
#include "alsutils/types.hpp"
#include <mpi.h>
#include <array>

namespace alsutils {
namespace mpi {

//! Convenience function to do the type cast from int to whatever MPI_Offset is (usually long long int)
std::array<MPI_Offset, 3> to_mpi_offset(const ivec3& integerVector) {
    std::array<MPI_Offset, 3> converted;

    converted[0] = MPI_Offset(integerVector.x);
    converted[1] = MPI_Offset(integerVector.y);
    converted[2] = MPI_Offset(integerVector.z);

    return converted;
}
}
}
