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

#include "alsutils/get_standard_c_library.hpp"
#include <boost/predef.h>
#if BOOST_OS_LINUX
#include <gnu/libc-version.h>
#endif

namespace alsutils {
std::string getStandardCLibrary() {
#if BOOST_OS_LINUX
    return std::string("GNU libc ") + std::string(gnu_get_libc_version())
            + std::string(" ") + std::string(gnu_get_libc_release());
#else
    return "Unknown standard C library";
#endif
}
}
