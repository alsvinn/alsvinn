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

#include "alsfvm/io/io_utils.hpp"
#include <sstream>

namespace alsfvm {
namespace io {


///
/// \brief getOutputname creates the output filename
/// \note This does not include an extension
/// \param filename the base filename (eg. "simulation")
/// \param snapshotNumber the current snapshot number (this not the timestep
///        number)
/// \note snapshotNumber is essentially the number of snapshots that has been
///                      saved up until now.
/// \return the filename with extra timestep information attached.
///
std::string getOutputname(const std::string& filename,
    size_t snapshotNumber) {
    std::stringstream ss;
    ss << filename << "_" << snapshotNumber;

    return ss.str();
}
}
}
