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
#include "alsutils/error/Exception.hpp"
#include <netcdf.h>

#define NETCDF_SAFE_CALL(x) {\
    auto error = x; \
    if (error) { \
        THROW("NetCDF error in call to\n\t" << #x << "\n\nError code: " << error \
            <<"\n\nError message: " << nc_strerror(error)); \
    } \
}

namespace alsfvm {
namespace io {
typedef int netcdf_raw_ptr;



template<class RealType>
struct NetCDFType {
    using type = double;
};


template<>
struct NetCDFType<float> {
    using type = float;
};

//! Gets the type corresponding to the alsfvm::real type
inline netcdf_raw_ptr getNetcdfRealType() {
    if (std::is_same<NetCDFType<real>::type, double>::value) {
        return NC_DOUBLE;
    } else if ((std::is_same<NetCDFType<real>::type, float>::value)) {
        return NC_FLOAT;
    }

    return NC_DOUBLE;
}

//! Wrapper function for nc_put_var_double
template<class RealType>
typename std::enable_if<std::is_same<RealType, double>::value, int>::type
nc_put_var_real(
    int ncid, int varid, const RealType* op) {
    return nc_put_var_double(ncid, varid, op);

}

//! Wrapper function for nc_put_var_double
template<class RealType>
typename std::enable_if<std::is_same<RealType, float>::value, int>::type
nc_put_var_real(
    int ncid, int varid, const RealType* op) {
    return nc_put_var_float(ncid, varid, op);

}

}

}


