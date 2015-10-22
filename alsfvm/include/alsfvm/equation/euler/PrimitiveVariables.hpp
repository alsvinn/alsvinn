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
	__device__ __host__ PrimitiveVariables()
        :rho(0), u(0, 0, 0), p(0)
    {
         // empty
    }

	__device__ __host__ PrimitiveVariables(real rho, real ux, real uy, real uz, real p)
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
///
/// Computes the component difference
/// \note Makes a new instance
///
__device__ __host__ inline PrimitiveVariables operator-(const PrimitiveVariables& a, const PrimitiveVariables& b) {
    return PrimitiveVariables(a.rho - b.rho, a.u.x - b.u.x, a.u.y - b.u.y, a.u.z - b.u.z, a.p - b.p);
}

///
/// Computes the component addition
/// \note Makes a new instance
///
__device__ __host__ inline PrimitiveVariables operator+(const PrimitiveVariables& a, const PrimitiveVariables& b) {
    return PrimitiveVariables(a.rho + b.rho, a.u.x + b.u.x, a.u.y + b.u.y, a.u.z + b.u.z, a.p + b.p);
}

///
/// Computes the product of a and b (scalar times vector)
/// \note Makes a new instance
////
__device__ __host__ inline PrimitiveVariables operator*(real a, const PrimitiveVariables& b) {
    return PrimitiveVariables(a*b.rho, a*b.u.x, a*b.u.y, a*b.u.z, a*b.p);
}

///
/// Computes the division of a by b
/// \note Makes a new instance
////
__device__ __host__ inline PrimitiveVariables operator/(const PrimitiveVariables& a, real b) {
    return PrimitiveVariables(a.rho / b, a.u.x / b, a.u.y / b, a.u.z / b, a.p / b);
}
}}}
