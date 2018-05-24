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

#include <cmath>
#include <limits>
#define ALSFVM_WENO_EPSILON 1e-8
namespace alsfvm {
namespace reconstruction {

template<int k>
class WENOCoefficients {
public:
    static const real epsilon;
    static real coefficients[];

    template<size_t index, class T>
    __device__ __host__ static real computeBeta(const T& stencil);


};

template<>
template<size_t index, class T>
__device__ __host__  real WENOCoefficients<2>::computeBeta(const T& V) {
    if (index == 0) {
        real beta = V[2] - V[1];
        return beta * beta;
    } else if (index == 1) {
        real beta = V[1] - V[0];
        return beta * beta;
    }

    static_assert(index < 2, "Only up to index 1 for order 2 in WENO");
    return 0;
}

template<>
template<size_t index, class T>
__device__ __host__ real WENOCoefficients<3>::computeBeta(const T& V) {
    if (index == 0) {
        return  13.0 / 12.0 * pow(V[2] - 2 * V[3] + V[4],
                2) + 1 / 4.0 * pow(3 * V[2] - 4 * V[3] + V[4], 2);
    } else if (index == 1) {
        return  13.0 / 12.0 * pow(V[1] - 2 * V[2] + V[3],
                2) + 1 / 4.0 * pow(V[1] -  V[3], 2);
    } else if (index == 2) {
        return  13.0 / 12.0 * pow(V[0] - 2 * V[1] + V[2],
                2) + 1 / 4.0 * pow(V[0] - 4 * V[1] + 3 * V[2], 2);
    }

    static_assert(index < 3, "Only up to index 1 for order 2 in WENO");
    return 0;
}







} // namespace alsfvm
} // namespace reconstruction
