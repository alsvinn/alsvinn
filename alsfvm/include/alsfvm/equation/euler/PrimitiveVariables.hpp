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

namespace alsfvm {
namespace equation {
namespace euler {
///
/// The holder struct for all relevant variables for the euler flux
/// These are supposed to be the primitive variables, ie. the variables
/// you would specify for eg. initial conditions.
///
template<int nsd>
class PrimitiveVariables {
public:
    typedef typename Types<nsd>::rvec rvec;


    __device__ __host__ PrimitiveVariables()
        : rho(0), u(0), p(0) {
        // empty
    }

    __device__ __host__ PrimitiveVariables(real rho, rvec u, real p)
        : rho(rho), u(u), p(p) {
        // empty
    }

    template<class T>
    __device__ __host__ PrimitiveVariables(T rho, T ux, T uy, T uz, T p)
        : rho(rho), u(rvec3{ux, uy, uz}), p(p) {
        static_assert(nsd == 3 || sizeof(T) == 0, "Only for 3 dimensions!");
    }

    template<class T>
    __device__ __host__ PrimitiveVariables(T rho, T ux, T uy, T p)
        : rho(rho), u(rvec2{ux, uy}), p(p) {
        static_assert(nsd == 2 || sizeof(T) == 0, "Only for 3 dimensions!");
    }

    ///
    /// \brief rho is the density
    ///
    real rho;

    ///
    /// \brief u is the velocity
    ///
    rvec u;

    ///
    /// \brief p is the pressure
    ///
    real p;

};
///
/// Computes the component difference
/// \note Makes a new instance
///
template<int nsd>
__device__ __host__ inline PrimitiveVariables<nsd> operator-
(const PrimitiveVariables<nsd>& a, const PrimitiveVariables<nsd>& b) {
    return PrimitiveVariables<nsd>(a.rho - b.rho, a.u - b.u, a.p - b.p);
}

///
/// Computes the component addition
/// \note Makes a new instance
///
template<int nsd>
__device__ __host__ inline PrimitiveVariables<nsd> operator+
(const PrimitiveVariables<nsd>& a, const PrimitiveVariables<nsd>& b) {
    return PrimitiveVariables<nsd>(a.rho + b.rho, a.u + b.u, a.p + b.p);
}

///
/// Computes the product of a and b (scalar times vector)
/// \note Makes a new instance
///
template<int nsd>
__device__ __host__ inline PrimitiveVariables<nsd> operator*(real a,
    const PrimitiveVariables<nsd>& b) {
    return PrimitiveVariables<nsd>(a * b.rho, a * b.u, a * b.p);
}

///
/// Computes the division of a by b
/// \note Makes a new instance
///
template<int nsd>
__device__ __host__ inline PrimitiveVariables<nsd> operator/
(const PrimitiveVariables<nsd>& a, real b) {
    return PrimitiveVariables<nsd>(a.rho / b, a.u / b, a.p / b);
}
}
}
}
