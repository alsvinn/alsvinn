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
#include <type_traits>
#include "alsutils/types.hpp"

//! Really easy way of computing the legendre polynomials
//! This file was only implemented since boost::legendre did not support cuda
//! if this becomes the case in the future, this file can be removed
//!
//! I have tried to keep the  interface 100% identical to boost, see
//! http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!
namespace alsutils {
namespace math {


//! Should compute the legendre polynomial of degree n at x
//! @param x the point to evuluate in
//!
//! See also http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!
//! @note This function is only implemented to get a GPU version of the legendre polynomials
//!       (as of version 1.66, boost still doens't have a gpu friendly legendre implementation)
template<int n>
__device__ __host__
real legendre(real x) {
  return real(1.0 / n) * (x * legendre < n - 1 > (x) - (n - 1) * legendre < n - 2 >
            (x));

}

//! Should compute the legendre polynomial of degree 1 at x
//! @param x the point to evuluate in
//!
//! See also http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!
//! @note This function is only implemented to get a GPU version of the legendre polynomials
//!       (as of version 1.66, boost still doens't have a gpu friendly legendre implementation)
template<>
__device__ __host__ real legendre<1>(real x) {
    return x;
}

//! Should compute the legendre polynomial of degree 0 at x
//! @param x the point to evuluate in
//!
//! See also http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!
//! @note This function is only implemented to get a GPU version of the legendre polynomials
//!       (as of version 1.66, boost still doens't have a gpu friendly legendre implementation)
template<>
__device__ __host__ real legendre<0>(real x) {
    return 1;
}

//! Should compute the legendre polynomial of degree degree at x
//! @param degree the degree of the polynomial
//! @param x the point to evuluate in
//!
//! See also http://www.boost.org/doc/libs/1_46_1/libs/math/doc/sf_and_dist/html/math_toolkit/special/sf_poly/legendre.html
//!
//! @note This function is only implemented to get a GPU version of the legendre polynomials
//!       (as of version 1.66, boost still doens't have a gpu friendly legendre implementation)
__device__ __host__ real legendre_p(int degree, real x) {
    switch (degree) {
    case 0:
        return legendre<0>(x);

    case 1:
        return legendre<1>(x);

    case 2:
        return legendre<2>(x);

    case 3:
        return legendre<3>(x);

    case 4:
        return legendre<4>(x);

    case 5:
        return legendre<5>(x);

    case 6:
        return legendre<6>(x);

    case 7:
        return legendre<7>(x);

    case 8:
        return legendre<8>(x);

    case 9:
        return legendre<9>(x);

    case 10:
        return legendre<10>(x);

    case 11:
        return legendre<11>(x);

    default:
        assert(false);
    }

    return 0;
}
}
}
