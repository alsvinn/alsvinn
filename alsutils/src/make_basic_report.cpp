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

#include "alsutils/config.hpp"
#include "alsutils/make_basic_report.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "alsutils/config.hpp"
#include "alsutils/get_boost_properties.hpp"
#include "alsutils/get_os_name.hpp"
#include "alsutils/get_standard_c_library.hpp"
#include "alsutils/get_hostname.hpp"
#include "alsutils/get_cpu_name.hpp"
#include "alsutils/get_username.hpp"
#include "alsutils/get_hostname.hpp"
#include "alsutils/mpi/get_mpi_version.hpp"
#include "alsutils/io/TextFileCache.hpp"
#include <boost/filesystem.hpp>

#include "alsutils/get_python_version.hpp"
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
boost::property_tree::ptree makeBasicReport() {
    boost::property_tree::ptree propertyTree;

    propertyTree.put("report.software", "alsvinn https://github.com/alsvinn");
    propertyTree.put("report.softwareVersion", getAlsvinnVersion());

    boost::filesystem::path currentWorkingDirectory(
        boost::filesystem::current_path());

    propertyTree.put("report.currentWorkingDirectory",
        currentWorkingDirectory.string());

    propertyTree.put("report.operatingSystem", alsutils::getOSName());
    propertyTree.put("report.username", alsutils::getUsername());
    propertyTree.put("report.host", alsutils::getHostname());
    propertyTree.put("report.standardCLibrary", alsutils::getStandardCLibrary());


    propertyTree.put("report.generatedAt", boost::posix_time::to_iso_string(
            boost::posix_time::second_clock::local_time()));

    propertyTree.put("report.CPU", alsutils::getCPUName());
    propertyTree.put("report.revision", getVersionControlID());
    propertyTree.put("report.versionControlStatus", getVersionControlStatus());
    propertyTree.put("report.buildType", getBuildType());
    propertyTree.put("report.cxxFlags", getCXXFlags());
    propertyTree.put("report.cxxFlagsDebug", getCXXFlagsDebug());
    propertyTree.put("report.cxxFlagsRelease", getCXXFlagsRelease());
    propertyTree.put("report.cxxFlagsMinSizeRel", getCXXFlagsMinSizeRel());
    propertyTree.put("report.cxxFlagsRelWithDebInfo", getCXXFlagsRelWithDebInfo());
    propertyTree.put("report.cudaFlags", getCUDAFlags());
    propertyTree.put("report.cudaVersion", getCUDAVersion());


    propertyTree.put("report.cxxCompiler", getCompilerName());
    propertyTree.put("report.cudaCompiler", getCUDACompilerName());
    propertyTree.add_child("report.boost", alsutils::getBoostProperties());

    // Floating point stuff
    propertyTree.put("report.floatingPointPrecisionDescription",
        getFloatingPointPrecisionDescription());
    propertyTree.put("report.floatingPointType", getFloatingPointType());

    propertyTree.put("report.floatingPointMax", getFloatingPointMaxValue());
    propertyTree.put("report.floatingPointMin", getFloatingPointMinValue());
    propertyTree.put("report.floatingPointEpsilon", getFloatingPointEpsilon());



#ifdef ALSVINN_HAVE_CUDA
    propertyTree.add_child("report.cudaProperties",
        alsutils::cuda::getDeviceProperties());
#endif

#ifdef ALSVINN_USE_MPI
    int mpiNumThreads = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiNumThreads);
    propertyTree.put("report.mpiEnabled", true);
    propertyTree.put("report.mpiProcesses", mpiNumThreads);
    propertyTree.put("report.mpiVersion", alsutils::mpi::getMPIVersion());
#else
    propertyTree.put("report.mpiEnabled", false);
    propertyTree.put("report.mpiProcesses", 1);
#endif

#ifdef _OPENMP
    propertyTree.put("report.ompEnabled", true);
    propertyTree.put("report.ompThreads", omp_get_max_threads());
#else
    propertyTree.put("report.ompEnabled", false);
    propertyTree.put("report.ompThreads", 1);
#endif


    auto& textCache = alsutils::io::TextFileCache::getInstance();

    for (const auto& filename : textCache.getAllLoadedFiles()) {
        // We need to remove dashes in filename
        auto filenameWithoutDashes = boost::algorithm::replace_all_copy(filename, "/",
                "_dash_");
        auto filenameWithEscapedDots = boost::algorithm::replace_all_copy(
                filenameWithoutDashes, ".",
                "_DOT_");
        propertyTree.put("report.loadedTextFiles." + filenameWithEscapedDots,
            textCache.loadTextFile(filename));
    }


    propertyTree.put("report.pythonVersion", getPythonVersion());

    return propertyTree;
}
}
