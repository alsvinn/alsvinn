#pragma once
#include "alsfvm/io/netcdf_utils.hpp"

namespace alsfvm {
namespace io {
    void parallelNetcdfWriteReport(netcdf_raw_ptr file);
}
}
