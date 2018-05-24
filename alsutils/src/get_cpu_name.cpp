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
