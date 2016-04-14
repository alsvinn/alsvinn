#pragma once
#include <string>
#include "alsfvm/types.hpp"

namespace alsfvm { namespace numflux { 



//! The central flux.
//!
//! Approximate the flux through
//!
//! \f[
//! \frac{u_{j+1/2}-u_{j-1/2}}{2}}
//! \f]
//!
template<class Equation>
class LaxFriedrichs {
public:
    ///
    /// \brief name is "laxfriedrichs"
    ///
    static const std::string name;

    template<int direction>
    __device__ __host__ inline static real computeFlux(const Equation& eq,
                                                       const typename Equation::AllVariables& left,
                                                       const typename Equation::AllVariables& right,
                                                       typename Equation::ConservedVariables& F)
    {
//F =  0.5*(eq.f(Ul,d) + o.f(Ur,d)) - 0.5*(dx/dt).*(Ur-Ul);;

        // This looks a bit weird, but it is OK. The basic principle is that AllVariables
        // is both a conservedVariable and an extra variable, hence we need to pass
        // it twice since this function expects both.
        return fmax(eq.template computeWaveSpeed<direction>(left, left),
                        eq.template computeWaveSpeed<direction>(right, right));
    }
};
} // namespace alsfvm
} // namespace numflux
