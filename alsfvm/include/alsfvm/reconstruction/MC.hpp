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
#include "alsfvm/reconstruction/minmod.hpp"

namespace alsfvm {
namespace reconstruction {

template<class Equation>
class MC {
public:


    __device__ __host__ static void reconstruct(Equation eq,
        typename Equation::ConstViews& in,
        size_t x, size_t y, size_t z,
        typename Equation::Views& leftView,
        typename Equation::Views& rightView,
        bool xDir, bool yDir, bool zDir) {
        const size_t indexOut = leftView.index(x, y, z);
        const size_t indexRight = leftView.index(x + xDir, y + yDir, z + zDir);
        const size_t indexLeft = leftView.index(x - xDir, y - yDir, z - zDir);

        for (size_t var = 0; var < Equation::getNumberOfConservedVariables(); ++var) {
            const real left = in.get(var).at(indexLeft);
            const real middle = in.get(var).at(indexOut);
            const real right = in.get(var).at(indexRight);

            const real sigma = minmod(2 * (right - middle),
                    (right - left) / 2,
                    2 * (middle - left));

            leftView.get(var).at(indexOut) = middle - sigma / 2;
            rightView.get(var).at(indexOut) = middle + sigma / 2;
        }

    }

    __device__ __host__ static size_t getNumberOfGhostCells() {
        return 2;
    }
};
} // namespace reconstruction
} // namespace alsfvm
