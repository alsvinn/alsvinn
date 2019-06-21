#pragma once
#include <pnetcdf.h>
#include "alsfvm/types.hpp"
#include "alsfvm/io/netcdf_utils.hpp"

namespace alsfvm {
namespace io {

//! Wrapper function for ncmpi_put_vara_double_all
template<class RealType>
typename std::enable_if<std::is_same<RealType, double>::value, int>::type ncmpi_put_vara_real_all(
    int ncid, int varid, const MPI_Offset* start,
    const MPI_Offset* count, const RealType* op) {
    return ncmpi_put_vara_double_all(ncid, varid, start, count, op);

}

//! Wrapper function for ncmpi_put_vara_float_all
template<class RealType>
typename std::enable_if<std::is_same<RealType, float>::value, int>::type ncmpi_put_vara_real_all(
    int ncid, int varid, const MPI_Offset* start,
    const MPI_Offset* count, const RealType* op) {
    return ncmpi_put_vara_float_all(ncid, varid, start, count, op);

}

}
}
