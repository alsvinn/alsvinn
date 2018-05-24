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
#include "alsfvm/types.hpp"
#include "alsfvm/memory/Memory.hpp"

namespace alsfvm {
namespace functional {

__device__ __host__ inline int makePositive(int position, int N) {
    if (position < 0) {
        position += N;
    }

    return position;

}

__device__ __host__ inline void computeStructureCube(alsfvm::memory::View<real>&
    output,
    const alsfvm::memory::View<const real>& input,
    int i, int j, int k, int h, int nx, int ny, int nz,
    int ngx, int ngy, int ngz, int dimensions, real p) {
    const auto u = input.at(i + ngx, j + ngy, k + ngz);

    for (int d = 0; d < dimensions; d++) {
        // side = 0 represents bottom, side = 1 represents top
        for (int side = 0; side < 2; side++) {
            const bool zDir = (d == 2);
            const bool yDir = (d == 1);
            const bool xDir = (d == 0);
            // Either we start on the left (i == 0), or on the right(i==1)
            const int zStart = zDir ?
                (side == 0 ? k - h + 1 : k + h) : (dimensions > 2 ? k - h + 1 : 0);

            const int zEnd = zDir ?
                (zStart + 1) : (dimensions > 2 ? k + h : 1);

            const int yStart = yDir ?
                (side == 0 ? j - h : j + h + 1) : (dimensions > 1 ? j - h + 1 : 0);

            const int yEnd = yDir ?
                (yStart + 1) : (dimensions > 1 ? j + h : 1);

            const int xStart = xDir ?
                (side == 0 ? i - h : i + h + 1) : i - h;

            const int xEnd = xDir ?
                (xStart + 1) : i + h + 1;

            for (int z = zStart; z < zEnd; z++) {
                for (int y = yStart; y < yEnd; y++) {
                    for (int x = xStart; x < xEnd; x++) {
                        const auto u_h = input.at(makePositive(x, nx) % nx + ngx,
                                makePositive(y, ny) % ny + ngy,
                                makePositive(z, nz) % nz + ngz);
                        output.at(h) += pow(fabs(u_h - u), p) / (nx * ny * nz);
                    }
                }
            }
        }
    }
}
}
}
