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
/// These are supposed to be the extra variables (non-conserved)
///
template<int nsd>
class ExtraVariables {
public:

    typedef typename Types<nsd>::rvec rvec;


    __device__ __host__ ExtraVariables(real p, rvec u)
        : p(p), u(u) {

    }

    template<class T>
    __device__ __host__ ExtraVariables(T p, T ux, T uy, T uz)
        : p(p), u(rvec3{ux, uy, uz}) {
        static_assert(nsd == 3 || sizeof(T) == 0, "Only for 3 dimensions!");
    }

    template<class T>
    __device__ __host__ ExtraVariables(T p, T ux, T uy)
        : p(p), u(rvec2{ux, uy}) {
        static_assert(nsd == 2 || sizeof(T) == 0, "Only for 3 dimensions!");
    }

    __device__ __host__ ExtraVariables()
        : p(0), u(0) {

    }


    real p;
    rvec u;
};


} // namespace alsfvm

} // namespace numflux

} // namespace euler

