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
#include "alsfvm/reconstruction/WENOCoefficients.hpp"

namespace alsfvm {
namespace reconstruction {

///
/// Simple WENO2 reconstruction. This is to be used with the ReconstructionCPU
/// or ReconstructionCUDA class!
///
template<class Equation>
class WENO2 {
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
        const real i0 = eq.getWeight(in, indexOut);
        const real b0 = square(eq.getWeight(in, indexRight) - i0);
        const real b1 = square(i0 - eq.getWeight(in, indexLeft));

        const real a0Left = 1 / (3 * (ALSFVM_WENO_EPSILON + b0) *
                (ALSFVM_WENO_EPSILON + b0));
        const real a1Left = 2 / (3 * (ALSFVM_WENO_EPSILON + b1) *
                (ALSFVM_WENO_EPSILON + b1));
        const real w0Left = a0Left / (a0Left + a1Left);
        const real w1Left = a1Left / (a0Left + a1Left);

        const real a0Right = 2 / (3 * (ALSFVM_WENO_EPSILON + b0) *
                (ALSFVM_WENO_EPSILON + b0));
        const real a1Right = 1 / (3 * (ALSFVM_WENO_EPSILON + b1) *
                (ALSFVM_WENO_EPSILON + b1));
        const real w0Right = a0Right / (a0Right + a1Right);
        const real w1Right = a1Right / (a0Right + a1Right);
#if 1
#ifdef __CUDA_ARCH__
#pragma unroll
#endif

        for (size_t var = 0; var < Equation::getNumberOfConservedVariables(); ++var) {
            leftView.get(var).at(indexOut) = 0.5 * (w1Left * in.get(var).at(indexLeft) +
                    (3 * w0Left + w1Left) * in.get(var).at(indexOut) -
                    w0Left * in.get(var).at(indexRight));

            rightView.get(var).at(indexOut) = 0.5 * (w0Right * in.get(var).at(indexRight) +
                    (3 * w1Right + w0Right) * in.get(var).at(indexOut) -
                    w1Right * in.get(var).at(indexLeft));

        }

#else
        typename Equation::ConservedVariables inLeft = eq.fetchConservedVariables(in,
                indexLeft);
        typename Equation::ConservedVariables inMiddle = eq.fetchConservedVariables(in,
                indexOut);
        typename Equation::ConservedVariables inRight = eq.fetchConservedVariables(in,
                indexRight);

        typename Equation::ConservedVariables left =  0.5 * (w1Left * inLeft +
                (3 * w0Left + w1Left) * inMiddle -
                w0Left * inRight);

        typename Equation::ConservedVariables right =  0.5 * (w0Right * inRight +
                (3 * w1Right + w0Right) * inMiddle -
                w1Right * inLeft);
        eq.setViewAt(leftView, indexOut, left);
        eq.setViewAt(rightView, indexOut, right);
#endif
    }

    __device__ __host__ static int getNumberOfGhostCells() {
        return 2;
    }
};
} // namespace alsfvm
} // namespace reconstruction
