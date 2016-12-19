#pragma once
#include "alsutils/error/Exception.hpp"

#define NETCDF_SAFE_CALl(x) {\
    auto error = x; \
    if (error) { \
        THROW("NetCDF error in call to\n\t" << #x << "\n\nError code: " << error \
            <<"\n\nError message: " << nc_strerror(error)); \
    } \
}

namespace alsfvm {
    typedef int netcdf_raw_ptr;

}
