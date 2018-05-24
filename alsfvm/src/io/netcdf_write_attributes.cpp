
#include "alsfvm/io/parallel_netcdf_write_report.hpp"
#include <pnetcdf.h>
#include "alsutils/make_basic_report.hpp"
#include <iostream>
namespace alsfvm {
namespace io {

void netcdfWriteAttributes(netcdf_raw_ptr varid,
    const std::string& basename,
    const boost::property_tree::ptree& propertyTree) {

    if (propertyTree.empty()) {
        std::string data = propertyTree.data();
        NETCDF_SAFE_CALl(nc_put_att_text(varid, NC_GLOBAL, basename.c_str(),
                data.size(), data.c_str()));
    } else {
        for (auto& node : propertyTree) {
            std::string newName = basename + "." + node.first;

            netcdfWriteAttributes(varid, newName, node.second);
        }
    }


}

}
}
