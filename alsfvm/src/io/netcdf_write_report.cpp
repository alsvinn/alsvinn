
#include "alsfvm/io/netcdf_write_report.hpp"
#include "alsfvm/io/netcdf_write_attributes.hpp"
#include <pnetcdf.h>
#include "alsutils/make_basic_report.hpp"
#include <iostream>
namespace alsfvm {
namespace io {

void netcdfWriteReport(netcdf_raw_ptr file) {
    auto propertyTree = alsutils::makeBasicReport();
    netcdfWriteAttributes(file,  "alsvinn_report",
        propertyTree.get_child("report"));
}
}
}
