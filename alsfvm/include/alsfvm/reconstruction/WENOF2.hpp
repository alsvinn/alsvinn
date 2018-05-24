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
#include "alsfvm/equation/euler/Euler.hpp"
#include <algorithm>

namespace alsfvm {
namespace reconstruction {
///
/// Simple WENO2 reconstruction. This is to be used with the ReconstructionCPU
/// or ReconstructionCUDA class!
///
/// This WENO2 implementation enforces the equation based constraints by computing
/// the reconstruction in the primitive variables and enforcing the natural bounds there.
///
template<class Equation>
class WENOF2 {
public:
    __device__ __host__ static void reconstruct(Equation eq,
        typename Equation::ConstViews& in,
        size_t x, size_t y, size_t z,
        typename Equation::Views& leftView,
        typename Equation::Views& rightView,
        bool xDir, bool yDir, bool zDir) {
        const real BOUND = 1.9;
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

        typename Equation::PrimitiveVariables inLeft =
            eq.computePrimitiveVariables(eq.fetchConservedVariables(in, indexLeft));

        typename Equation::PrimitiveVariables inMiddle =
            eq.computePrimitiveVariables(eq.fetchConservedVariables(in, indexOut));

        typename Equation::PrimitiveVariables inRight =
            eq.computePrimitiveVariables(eq.fetchConservedVariables(in, indexRight));

        typename Equation::PrimitiveVariables dPLeft = w1Left * (inMiddle - inLeft) +
            w0Left * (inRight - inMiddle);


        dPLeft.rho = fmax(-BOUND * inMiddle.rho, fmin(BOUND * inMiddle.rho,
                    dPLeft.rho));
        dPLeft.p = fmax(-BOUND * inMiddle.p, fmin(BOUND * inMiddle.p, dPLeft.p));

        real LLLeft = 0.125 * inMiddle.rho * (dPLeft.u.dot(dPLeft.u)) -
            0.5 * fmin(0.0, dPLeft.rho * (inMiddle.u.dot(dPLeft.u))) +
            0.5 * dPLeft.rho * dPLeft.rho * dPLeft.u.dot(dPLeft.u) / inMiddle.rho;


        real R = inMiddle.p / (eq.getGamma() - 1);
        real aijkLeft = 0.5 * std::sqrt(R / fmax(R, LLLeft));





        typename Equation::ConservedVariables left = eq.computeConserved(
                inMiddle - aijkLeft * dPLeft);


        typename Equation::PrimitiveVariables dPRight = w1Right * (inMiddle - inLeft) +
            w0Right * (inRight - inMiddle);


        dPRight.rho = fmax(-BOUND * inMiddle.rho, fmin(BOUND * inMiddle.rho,
                    dPRight.rho));
        dPRight.p = fmax(-BOUND * inMiddle.p, fmin(BOUND * inMiddle.p, dPRight.p));

        real LLRight = 0.125 * inMiddle.rho * (dPRight.u.dot(dPRight.u)) -
            0.5 * fmin(0.0, dPRight.rho * (inMiddle.u.dot(dPRight.u))) +
            0.5 * dPRight.rho * dPRight.rho * dPRight.u.dot(dPRight.u) / inMiddle.rho;

        real aijkRight = 0.5 * std::sqrt(R / fmax(R, LLRight));

        typename Equation::ConservedVariables right =  eq.computeConserved(
                inMiddle + aijkRight * dPRight);
        eq.setViewAt(leftView, indexOut, left);
        eq.setViewAt(rightView, indexOut, right);

    }

    __device__ __host__ static int getNumberOfGhostCells() {
        return 2;
    }
};
} // namespace alsfvm
} // namespace reconstruction
