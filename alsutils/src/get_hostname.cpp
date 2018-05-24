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

#include "alsutils/get_hostname.hpp"
#include <boost/predef.h>
#include <vector>

#if BOOST_OS_LINUX
    #include <unistd.h>
#endif
namespace alsutils {
std::string getHostname() {
#if BOOST_OS_LINUX
    std::vector<char> buffer(32 * 1024, 0);

    auto returnValue = gethostname(buffer.data(), buffer.size());

    if (returnValue == 0) {
        return std::string(buffer.data());
    }

#endif
    return "Unknown hostname";
}
}
