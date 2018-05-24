#pragma once
#include <boost/property_tree/ptree.hpp>
namespace alsfvm {
namespace io {

void netcdfWriteAttributes(netcdf_raw_ptr varid,
    const std::string& basename,
    const boost::property_tree::ptree& propertyTree);
}
}
