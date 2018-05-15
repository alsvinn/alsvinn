
#include "alsfvm/io/parallel_netcdf_write_report.hpp"
#include <pnetcdf.h>
#include "alsutils/make_basic_report.hpp"
#include <iostream>
namespace alsfvm {
namespace io {

namespace {
void writePropertyTree(netcdf_raw_ptr varid,
    const boost::property_tree::ptree& propertyTree,
    const std::string& basename) {

    if (propertyTree.empty()) {
        std::string data = propertyTree.data();
        NETCDF_SAFE_CALl(nc_put_att_text(varid, NC_GLOBAL, basename.c_str(),
                data.size(), data.c_str()));
    } else {
        for (auto& node : propertyTree) {
            std::string newName = basename + "." + node.first;

            writePropertyTree(varid, node.second, newName);
        }
    }


}
}
void netcdfWriteReport(netcdf_raw_ptr file) {
    auto propertyTree = alsutils::makeBasicReport();
    writePropertyTree(file, propertyTree.get_child("report"), "alsvinn_report");
}
}
}
