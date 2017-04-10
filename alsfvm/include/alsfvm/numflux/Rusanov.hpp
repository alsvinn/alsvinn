#pragma once
#include "alsfvm/types.hpp"

#include <algorithm>
#include <cmath>

namespace alsfvm { namespace numflux {

    //! The Rusanov flux.
    //!
    //! Approximate the flux through
    //!
    //! \f[ \frac{u_{j+1/2}+u_{j-1/2}}{2}\f]
    //!
    template<class Equation>
    class Rusanov {
    public:
        ///
        /// \brief name is "rusanov"
        ///
        static const std::string name;

        template<int direction>
        __device__ __host__ inline static real computeFlux(const Equation& eq,
                                                           const typename Equation::AllVariables& left,
                                                           const typename Equation::AllVariables& right,
                                                           typename Equation::ConservedVariables& F)
        {
            typename Equation::ConservedVariables fluxLeft;
            eq.template computePointFlux<direction>(left, fluxLeft);
            typename Equation::ConservedVariables fluxRight;
            eq.template computePointFlux<direction>(right, fluxRight);
            real eigenLeft = eq.template computeEigenValues<direction>(left);
            real eigenRight = eq.template computeEigenValues<direction>(right);

            real eigenMax = fmax(eigenLeft, eigenRight);
            F = 0.5 * (fluxLeft+fluxRight) - 0.5*eigenMax*(right.u-left.u);

            // This looks a bit weird, but it is OK. The basic principle is that AllVariables
            // is both a conservedVariable and an extra variable, hence we need to pass
            // it twice since this function expects both.
            return fmax(eq.template computeWaveSpeed<direction>(left, left),
                            eq.template computeWaveSpeed<direction>(right, right));
        }
    };
} // namespace alsfvm
} // namespace numflux
