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
#include <array>
#include <type_traits>
#include "alsfvm/types.hpp"

///
/// This file includes various utility functions for the numerical fluxes
///
namespace alsfvm {
namespace numflux {
//! SFINAE test, see http://stackoverflow.com/a/257382
template <typename T>
class has_stencil {
    typedef char one;
    typedef long two;

    template <typename C> static one test(decltype(&C::hasStencil));
    template <typename C> static two test(...);

public:
    enum { value = sizeof(test<T>(0)) == sizeof(char) };
};
// The following are two functions to get the
// stencil for a numerical flux. The default
// stencil is always "0,1", while some fluxes
// may have a larger stencil

//! Gets the stencil for numerical fluxes that have a stencil defined
template<class NumericalFluxType>

__device__ __host__  auto getStencil(NumericalFluxType)
-> typename
std::enable_if<has_stencil<NumericalFluxType>::value, decltype(NumericalFluxType::stencil())>::type {
    return NumericalFluxType::stencil();
}

//! Gets the default numerical stencil (0,1)
//! See http://stackoverflow.com/a/31860104 for how the overloading works
template<class NumericalFluxType>

__device__ __host__
typename std::enable_if < !has_stencil<NumericalFluxType>::value,
ivec2 >::type getStencil(NumericalFluxType) {
    return { 0, 1 };
}

template<class Flux, class Equation, size_t direction>
__device__ __host__ real computeFluxForStencil(const Equation& eq,
    ivec2 indices,
    typename Equation::ConstViews& left,
    typename Equation::ConstViews& right,
    typename Equation::ConservedVariables& out) {

    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);
    // This needs to be done with some smart template recursion

    // This is the value for j+1/2
    auto rightIndex = indices[1];
    auto middleIndex = indices[0];
    typename Equation::AllVariables rightJpHf = eq.fetchAllVariables(left,
            rightIndex);


    // This is the value for j+1/2
    typename Equation::AllVariables leftJpHf = eq.fetchAllVariables(right,
            middleIndex);



    // F(U_l, U_r)
    typename Equation::ConservedVariables fluxMiddleRight;
    real waveSpeed = Flux::template computeFlux<direction>(eq, leftJpHf, rightJpHf,
        fluxMiddleRight);

    out = fluxMiddleRight;
    return waveSpeed;
}

//! For higher order fluxes.
//! \note Here we assume the left input array equals the right one
template<class Flux, class Equation, size_t direction>
__device__ __host__ real computeFluxForStencil(const Equation& eq,
    ivec4 indices,
    typename Equation::ConstViews& left,
    typename Equation::ConstViews& right,
    typename Equation::ConservedVariables& out) {
    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

    typename Equation::AllVariables u0 = eq.fetchAllVariables(left, indices[0]);
    typename Equation::AllVariables u1 = eq.fetchAllVariables(left, indices[1]);
    typename Equation::AllVariables u2 = eq.fetchAllVariables(left, indices[2]);
    typename Equation::AllVariables u3 = eq.fetchAllVariables(left, indices[3]);

    //
    typename Equation::ConservedVariables fluxMiddleRight;
    real waveSpeed = Flux::template computeFlux<direction>(eq, u0, u1, u2, u3,
        fluxMiddleRight);

    out = fluxMiddleRight;
    return waveSpeed;
}

//! For higher order fluxes.
//! \note Here we assume the left input array equals the right one
template<class Flux, class Equation, size_t direction>
__device__ __host__ real computeFluxForStencil(const Equation& eq,
    ivec6 indices,
    typename Equation::ConstViews& left,
    typename Equation::ConstViews& right,
    typename Equation::ConservedVariables& out) {
    const ivec3 directionVector(direction == 0, direction == 1, direction == 2);

    typename Equation::AllVariables u0 = eq.fetchAllVariables(left, indices[0]);
    typename Equation::AllVariables u1 = eq.fetchAllVariables(left, indices[1]);
    typename Equation::AllVariables u2 = eq.fetchAllVariables(left, indices[2]);
    typename Equation::AllVariables u3 = eq.fetchAllVariables(left, indices[3]);
    typename Equation::AllVariables u4 = eq.fetchAllVariables(left, indices[4]);
    typename Equation::AllVariables u5 = eq.fetchAllVariables(left, indices[5]);

    //
    typename Equation::ConservedVariables fluxMiddleRight;
    real waveSpeed = Flux::template computeFlux<direction>(eq, u0, u1, u2, u3, u4,
        u5, fluxMiddleRight);

    out = fluxMiddleRight;
    return waveSpeed;
}

}
}
