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

#include "alsutils/write_run_report.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "alsutils/config.hpp"
#include "alsutils/get_boost_properties.hpp"
#include "alsutils/get_os_name.hpp"
#include "alsutils/get_cpu_name.hpp"
#include "alsutils/mpi/get_mpi_version.hpp"
#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>
#include "alsutils/make_basic_report.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include "alsutils/cuda/get_device_properties.hpp"
#endif
#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace alsutils {
void writeRunReport(const std::string& executable,
    const std::string& name,
    const int cpuDurationMs,
    const int wall,
    const int timesteps,
    const int argc,
    char** argv) {
    boost::property_tree::ptree propertyTree = alsutils::makeBasicReport();

    propertyTree.put("report.executable", executable);
    propertyTree.put("report.name", name);

    propertyTree.put("report.endTime",
        boost::posix_time::to_iso_string(
            boost::posix_time::second_clock::local_time()));
    propertyTree.put("report.cpuDuration", cpuDurationMs);
    propertyTree.put("report.cpuDurationHuman",
        boost::posix_time::to_simple_string(boost::posix_time::time_duration(0, 0,
                cpuDurationMs / 1000, 0)));
    propertyTree.put("report.wallTime", wall);
    propertyTree.put("report.wallTimeHuman",
        boost::posix_time::to_simple_string(boost::posix_time::time_duration(0, 0,
                wall / 1000, 0)));
    std::stringstream commandLine;

    for (int i = 0; i < argc; ++i) {
        commandLine << argv[i] << " ";
    }

    propertyTree.put("report.timesteps", timesteps);
    propertyTree.put("report.command", commandLine.str());




    boost::property_tree::write_json(executable + "_" + name + "_report.json",
        propertyTree);
    boost::property_tree::write_xml(executable + "_" + name + "_report.xml",
        propertyTree);
}
}
