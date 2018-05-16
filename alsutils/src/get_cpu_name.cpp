#include <fstream>
#include <boost/predef.h>
#include <vector>
#include "alsutils/get_cpu_name.hpp"
#include <boost/algorithm/string.hpp>

namespace alsutils {
std::string getCPUName() {
#if BOOST_OS_LINUX
    std::ifstream input("/proc/cpuinfo");

    if (!input) {
        return "Unknown CPU (failed to read /proc/cpuinfo)";
    }

    std::string line;
    while (std::getline(input, line)) {
        if (line.find("model name") != line.npos) {
            std::vector<std::string> splitted;
            boost::split(splitted, line, boost::is_any_of(":"));

            if (splitted.size() > 1) {
                return boost::trim_copy(splitted[1]);
            }
        }
    }
    return "Unknown CPU (failed parsing /proc/cpuinfo)";
#else
    return "Unknown CPU";
#endif


}
}
