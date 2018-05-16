#include "alsutils/make_basic_report.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "alsutils/config.hpp"
#include "alsutils/get_boost_properties.hpp"
#include "alsutils/get_os_name.hpp"
#include "alsutils/get_hostname.hpp"
#include "alsutils/get_cpu_name.hpp"
#include "alsutils/get_username.hpp"
#include "alsutils/get_hostname.hpp"
#include "alsutils/mpi/get_mpi_version.hpp"
#include <boost/filesystem.hpp>

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

    boost::filesystem::path currentWorkingDirectory(
        boost::filesystem::current_path());

    propertyTree.put("report.currentWorkingDirectory",
        currentWorkingDirectory.string());

    propertyTree.put("report.operatingSystem", alsutils::getOSName());
    propertyTree.put("report.username", alsutils::getUsername());
    propertyTree.put("report.host", alsutils::getHostname());


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


    return propertyTree;
}
}
