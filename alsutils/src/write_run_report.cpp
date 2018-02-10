#include "alsutils/write_run_report.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "alsutils/config.hpp"
#include <sstream>
#include <fstream>

namespace alsutils {
void writeRunReport(const std::string& executable,
                    const std::string& name,
                    const int cpuDurationMs,
                    const int wall,
                    const int argc,
                    char** argv) {
    boost::property_tree::ptree propertyTree;
    propertyTree.put("report.executable", executable);
    propertyTree.put("report.name", name);
    propertyTree.put("report.endTime", boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time()));
    propertyTree.put("report.cpuDuration", cpuDurationMs);
    propertyTree.put("report.cpuDurationHuman",
                     boost::posix_time::to_simple_string(boost::posix_time::time_duration(0, 0, cpuDurationMs/1000,0)));
    propertyTree.put("report.wallTime", wall);
    propertyTree.put("report.wallTimeHuman",
                     boost::posix_time::to_simple_string(boost::posix_time::time_duration(0, 0, wall/1000,0)));
    std::stringstream commandLine;

    for(int i = 0; i < argc; ++i) {
        commandLine << argv[i] << " ";
    }

    propertyTree.put("report.command", commandLine.str());
    propertyTree.put("report.revision", getVersionControlID());
    propertyTree.put("report.versionControlStatus", getVersionControlStatus());
    propertyTree.put("report.buildType", getBuildType());
    propertyTree.put("report.cxxFlags", getCXXFlags());
    propertyTree.put("report.cudaFlags", getCUDAFlags());
    propertyTree.put("report.cudaVersion", getCUDAVersion());



    boost::property_tree::write_json(executable + "_" + name + "_report.json", propertyTree);
    boost::property_tree::write_xml(executable + "_" + name + "_report.xml", propertyTree);
}
}
