#pragma once
#include "alsfvm/types.hpp"

#include <algorithm>
#include "alsfvm/equation/linear/Linear.hpp"
#include <cmath>
namespace alsfvm {
namespace numflux {
namespace linear {

//! The upwind flux.
//!
class Upwind {
public:
    using Equation = equation::linear::Linear;
    ///
    /// \brief name is "upwind"
    ///
    static const std::string name;

    template<int direction>
    __device__ __host__ inline static real computeFlux(const Equation& eq,
        const typename Equation::AllVariables& left,
        const typename Equation::AllVariables& right,
        typename Equation::ConservedVariables& F) {

        typename Equation::ConservedVariables fluxLeft;
        eq.template computePointFlux<direction>(left, fluxLeft);


        F = fluxLeft;

        // This looks a bit weird, but it is OK. The basic principle is that AllVariables
        // is both a conservedVariable and an extra variable, hence we need to pass
        // it twice since this function expects both.
        return fmax(eq.template computeWaveSpeed<direction>(left, left),
                eq.template computeWaveSpeed<direction>(right, right));
    }
};
} // namespace linear
} // namespace numflux
} // namespace alsfvm
