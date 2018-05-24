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


#include "alsfvm/io/parallel_netcdf_write_report.hpp"
#include <pnetcdf.h>
#include "alsutils/make_basic_report.hpp"
#include <iostream>
namespace alsfvm {
namespace io {

void parallelNetcdfWriteAttributes(netcdf_raw_ptr varid,
    const std::string& basename,
    const boost::property_tree::ptree& propertyTree) {

    if (propertyTree.empty()) {
        std::string data = propertyTree.data();
        NETCDF_SAFE_CALl(ncmpi_put_att_text(varid, NC_GLOBAL, basename.c_str(),
                data.size(), data.c_str()));
    } else {
        for (auto& node : propertyTree) {
            std::string newName = basename + "." + node.first;

            parallelNetcdfWriteAttributes(varid, newName, node.second);
        }
    }
}

}
}
