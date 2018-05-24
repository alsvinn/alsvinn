#pragma once
#include "alsfvm/io/netcdf_utils.hpp"
#include <boost/property_tree/ptree.hpp>
namespace alsfvm {
namespace io {
void parallelNetcdfWriteAttributes(netcdf_raw_ptr file,
    const std::string& attributeBaseName,
    const boost::property_tree::ptree& properties);
}
}
