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
#include "alsfvm/equation/cubic/Cubic.hpp"
#include <iostream>

#include <algorithm>

namespace alsfvm {
namespace numflux {
namespace cubic {

//! The Roe flux for the cubic equation.
//!
//!

class Roe {
public:
    ///
    /// \brief name is "Roe"
    ///
    static const std::string name;

    template<int direction>
    __device__ __host__ inline static real computeFlux(const
        equation::cubic::Cubic& eq,
        const equation::cubic::AllVariables& left,
        const equation::cubic::AllVariables& right,
        equation::cubic::ConservedVariables& F) {
        using namespace equation::cubic;

        ConservedVariables fluxLeft;
        eq.computePointFlux<direction>(AllVariables(fmax(left.u, real(0.0))), fluxLeft);

        ConservedVariables fluxRight;
        eq.computePointFlux<direction>(AllVariables(fmin(right.u, real(0.0))),
            fluxRight);

        real A = 0;

        if (left.u != right.u) {
            A = (fluxLeft.u - fluxRight.u) / (left.u - right.u);
        } else {
            A = 3 * left.u * left.u;
        }

        if (A >= 0) {
            F = fluxLeft;
        } else {
            F = fluxRight;
        }

        // This looks a bit weird, but it is OK. The basic principle is that AllVariables
        // is both a conservedVariable and an extra variable, hence we need to pass
        // it twice since this function expects both.
        return fmax(eq.computeWaveSpeed<direction>(left, left),
                eq.computeWaveSpeed<direction>(right, right));
    }
};
} // namespace alsfvm
} // namespace numflux
} // namespace cubic
