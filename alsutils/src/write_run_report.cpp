#include "alsutils/write_run_report.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "alsutils/config.hpp"
#include <sstream>
#include <fstream>
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
void writeRunReport(const std::string& executable,
    const std::string& name,
    const int cpuDurationMs,
    const int wall,
    const int timesteps,
    const int argc,
    char** argv) {
    boost::property_tree::ptree propertyTree;

    propertyTree.put("report.executable", executable);
    propertyTree.put("report.name", name);
    boost::filesystem::path currentWorkingDirectory(
        boost::filesystem::current_path());
    propertyTree.put("report.currentWorkingDirectory",
        currentWorkingDirectory.string());
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
    propertyTree.put("report.revision", getVersionControlID());
    propertyTree.put("report.versionControlStatus", getVersionControlStatus());
    propertyTree.put("report.buildType", getBuildType());
    propertyTree.put("report.cxxFlags", getCXXFlags());
    propertyTree.put("report.cudaFlags", getCUDAFlags());
    propertyTree.put("report.cudaVersion", getCUDAVersion());

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



    boost::property_tree::write_json(executable + "_" + name + "_report.json",
        propertyTree);
    boost::property_tree::write_xml(executable + "_" + name + "_report.xml",
        propertyTree);
}
}
