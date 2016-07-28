#pragma once
#include <string>
#include "alsfvm/types.hpp"

namespace alsfvm { namespace numflux { 



//! The Laxt Friedrich flux. NOT WORKING.
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
        static_assert(false, "Lax-Friedrich flux not implemented");
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
