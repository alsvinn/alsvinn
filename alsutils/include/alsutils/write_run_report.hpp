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

#pragma once
#include <string>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace alsutils {

//! Writes a run report to the json file
//!
//!    executable_name_report.json
//!
//! and to the xml file
//!
//!     executable_name_report.xml
//!
void writeRunReport(const std::string& executable,
    const std::string& name,
    const int cpuDurationMs,
    const int wall,
    const int timesteps,
    const int argc,
    char** argv);
}
