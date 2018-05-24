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
//! Various utility functions to implement the tecno flux

namespace alsfvm {
namespace numflux {
inline __device__ __host__ real bar(real left, real right) {
    return (left + right) / 2;
}

inline __device__ __host__ real diff(real left, real right) {
    return right - left;
}
inline __device__ __host__ real ln(real left, real right) {
    if (diff(left, right) == 0 || diff(log(left), log(right)) == 0) {
        return 0;
    }

    return diff(left, right) / diff(log(left), log(right));
}

//! Computes F/ln (left, right), but takes care if divLn==0,
//! then it returns 0
inline __device__ __host__ real divLn(real left, real right, real F) {
    auto div = ln(left, right);

    if (div == 0) {
        return 0;
    } else {
        return F / div;
    }
}

inline __device__ __host__ real div(real a, real b) {
    if (a == 0) {
        return 0;
    } else {
        return a / b;
    }
}
}
}