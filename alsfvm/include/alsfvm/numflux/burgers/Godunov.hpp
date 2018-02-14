#pragma once
#include "alsfvm/equation/burgers/Burgers.hpp"
#include <iostream>

#include <algorithm>

namespace alsfvm {
namespace numflux {
namespace burgers {

//! The Godunov flux for the Burgers equation.
//!
//! Approximate the flux through
//!
//! \f[ F_{i+1/2} = \max(f(\max(U_{j}, 0)), f(\min(U_{j+1},0))). \f]
//!

class Godunov {
public:
    ///
    /// \brief name is "godunov"
    ///
    static const std::string name;

    template<int direction>
    __device__ __host__ inline static real computeFlux(const
        equation::burgers::Burgers& eq,
        const equation::burgers::AllVariables& left,
        const equation::burgers::AllVariables& right,
        equation::burgers::ConservedVariables& F) {
        using namespace equation::burgers;
        ConservedVariables fluxLeft;
        eq.computePointFlux<direction>(AllVariables(fmax(left.u, real(0.0))), fluxLeft);
        ConservedVariables fluxRight;
        eq.computePointFlux<direction>(AllVariables(fmin(right.u, real(0.0))),
            fluxRight);

        F = ConservedVariables(fmax(fluxLeft.u, fluxRight.u));

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
