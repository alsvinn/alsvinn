#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm { namespace equation { namespace euler {
///
/// The holder struct for all relevant variables for the euler flux
/// These are supposed to be the primitive variables, ie. the variables
/// you would specify for eg. initial conditions.
///
class PrimitiveVariables {
public:
    PrimitiveVariables()
        :rho(0), u(0, 0, 0), p(0)
    {
         // empty
    }

    PrimitiveVariables(real rho, real ux, real uy, real uz, real p)
        : rho(rho), u(ux, uy, uz), p(p)
    {
        // empty
    }

    ///
    /// \brief rho is the density
    ///
    real rho;

    ///
    /// \brief u is the velocity
    ///
    rvec3 u;

    ///
    /// \brief p is the pressure
    ///
    real p;

};

}}}
