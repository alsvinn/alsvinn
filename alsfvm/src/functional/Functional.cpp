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

#include "alsfvm/functional/Functional.hpp"

namespace alsfvm {
namespace functional {

ivec3 Functional::getGhostCellSizes(const grid::Grid&,
    const volume::Volume&) const {
    return ivec3{0, 0, 0};
}

std::string Functional::getPlatformToAllocateOn(const std::string& platform)
const {
    return platform;
}

}
}
