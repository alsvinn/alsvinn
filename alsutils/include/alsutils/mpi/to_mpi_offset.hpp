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
