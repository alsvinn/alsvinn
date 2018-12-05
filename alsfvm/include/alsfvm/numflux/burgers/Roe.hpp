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
#include "alsfvm/equation/burgers/Burgers.hpp"
#include <iostream>

#include <algorithm>

namespace alsfvm {
namespace numflux {
namespace burgers {

//! The Roe flux for the Burgers equation.
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
        equation::burgers::Burgers& eq,
        const equation::burgers::AllVariables& left,
        const equation::burgers::AllVariables& right,
        equation::burgers::ConservedVariables& F) {
        using namespace equation::burgers;

        if (left.u >= -right.u) {
            ConservedVariables fluxLeft;
            eq.computePointFlux<direction>(AllVariables(fmax(left.u, real(0.0))), fluxLeft);
            F = fluxLeft;
        } else {
            ConservedVariables fluxRight;
            eq.computePointFlux<direction>(AllVariables(fmin(right.u, real(0.0))),
                fluxRight);
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
} // namespace burgers
