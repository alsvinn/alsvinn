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
#include "alsfvm/memory/View.hpp"

namespace alsfvm {
namespace boundary {

class Neumann {
public:
    __device__ __host__ static void applyBoundary(alsfvm::memory::View<real>&
        memoryArea,
        size_t x, size_t y, size_t z, size_t boundaryCell, size_t numberOfGhostCells,
        bool top, bool xDir, bool yDir, bool zDir) {
        const int sign = top ? -1 : 1;
        memoryArea.at(x - sign * boundaryCell * xDir,
            y - sign * boundaryCell * yDir,
            z - sign * boundaryCell * zDir )
            = memoryArea.at(x + sign * (boundaryCell - 1) * xDir,
                    y + sign * (boundaryCell - 1) * yDir,
                    z + sign * (boundaryCell - 1) * zDir);
    }

};
}
}
