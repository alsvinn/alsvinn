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
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include <cassert>
#include <cmath>
namespace alsfvm {
namespace equation {
namespace euler {

template<int nsd>
class AllVariables : public ConservedVariables<nsd>,
    public ExtraVariables<nsd> {
public:

    typedef typename Types<nsd>::rvec rvec;
    __device__ __host__ AllVariables(real rho, rvec m, real E, real p, rvec u) :
        ConservedVariables<nsd>(rho, m, E), ExtraVariables<nsd>(p, u) {
    }

    __device__ __host__ const ConservedVariables<nsd>& conserved() const {
        return *this;
    }

    __device__ __host__ AllVariables(const ConservedVariables<nsd>& conserved,
        const ExtraVariables<nsd>& extra)
        : ConservedVariables<nsd>(conserved), ExtraVariables<nsd>(extra) {
    }

};




} // namespace alsfvm
} // namespace numflux
} // namespace euler

