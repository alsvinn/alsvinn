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

//!
//! Simple class to hold the variables relevant for the Tecno
//! scheme, see definition of \f$\vec{z}\f$ in
//!
//! http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
//!
//! eq (6.11). That is, we set
//!
//! \f[\vec{z} = \left(\begin{array}{l} \sqrt{\frac{\rho}{p}}\\ \sqrt{\frac{\rho}{p}}u\\ \sqrt{\frac{\rho}{p}}v\\ \sqrt{\rho p}\end{array}\right).\f]
//!
template<int nsd>
class TecnoVariables {
public:
    typedef typename Types < nsd + 2 >::rvec state_vector;
    typedef typename Types<nsd>::rvec rvec;


    __device__ __host__ TecnoVariables(real z1, rvec zu, real z5);

    state_vector z;
};

template<>
__device__ __host__ inline TecnoVariables<1>::TecnoVariables(real z1, rvec1 zu,
    real z5)
    : z(z1, zu.x, z5) {
    // empty
}

template<>
__device__ __host__ inline TecnoVariables<2>::TecnoVariables(real z1, rvec2 zu,
    real z5)
    : z(z1, zu.x, zu.y, z5) {
    // empty
}


template<>
__device__ __host__ inline TecnoVariables<3>::TecnoVariables(real z1, rvec3 zu,
    real z5)
    : z(z1, zu.x, zu.y, zu.z, z5) {
    // empty
}

} // namespace alsfvm
} // namespace equation
} // namespace euler
