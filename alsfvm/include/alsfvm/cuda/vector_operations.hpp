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
///
/// Various vector operations in CUDA
///

namespace alsfvm {
namespace cuda {

///
/// Adds a and b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a or b). Must have length size (in T)
/// \param a must have length size (in T)
/// \param b must have length size (in T)
/// \param size the size of the memory (in T)
///
template<class T>
void add(T* result, const T* a, const T* b, size_t size);

///
/// Multiplies a and b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a or b). Must have length size (in T)
/// \param a must have length size (in T)
/// \param b must have length size (in T)
/// \param size the size of the memory (in T)
///
template<class T>
void multiply(T* result, const T* a, const T* b, size_t size);

///
/// Subtracts a from b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a or b). Must have length size (in T)
/// \param a must have length size (in T)
/// \param b must have length size (in T)
/// \param size the size of the memory (in T)
///
template<class T>
void subtract(T* result, const T* a, const T* b, size_t size);

///
/// Divides a by b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a or b). Must have length size (in T)
/// \param a must have length size (in T)
/// \param b must have length size (in T)
/// \param size the size of the memory (in T)
///
template<class T>
void divide(T* result, const T* a, const T* b, size_t size);

///
/// Adds scalar to each component of b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a ). Must have length size (in T)
/// \param a must have length size (in T)
/// \param scalar the scalar
/// \param size the size of the memory (in T)
///
template<class T>
void add(T* result, const T* a, T scalar, size_t size);

///
/// Multiplies scalar to each component of b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a ). Must have length size (in T)
/// \param a must have length size (in T)
/// \param size the size of the memory (in T)
/// \param scalar the scalar
///
template<class T>
void multiply(T* result, const T* a, T scalar, size_t size);

///
/// Subtracts scalar from each component of b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a ). Must have length size (in T)
/// \param a must have length size (in T)
/// \param scalar the scalar
/// \param size the size of the memory (in T)
///
template<class T>
void subtract(T* result, const T* a, T scalar, size_t size);

///
/// Divides scalar from each component of b and stores the result to result
/// \param result the device memory to write to (can be the same
///               as a ). Must have length size (in T)
/// \param a must have length size (in T)
/// \param scalar the scalar
/// \param size the size of the memory (in T)
///
template<class T>
void divide(T* result, const T* a, T scalar, size_t size);

//! Adds the memory with coefficients to this memory area
//! Here we compute the sum
//! \f[ v_1^{\mathrm{new}}=a_1v_1+a_2v_2+a_3v_3+a_4v_4+a_5v_5+a_6v_6\f]
//! where \f$v_1\f$ is the volume being operated on.
template<class T>
void add_linear_combination(T a1, T* v1,
    T a2, const T* v2,
    T a3, const T* v3,
    T a4, const T* v4,
    T a5, const T* v5,
    size_t size);

//! Basically runs
//!
//! \f[a += pow(b,power)\f]
template<class T>
void add_power(T* a, const T* b, double power, size_t size);


//! Basically runs
//!
//! \f[a += pow(b,power)\f]
template<class T>
void subtract_power(T* a, const T* b, double power, size_t size);


template<class T>
T compute_total_variation(const T* a, size_t nx, size_t ny, size_t nz, int p,
    const ivec3& start, const ivec3& end);

template<class T>
T compute_total_variation(const T* a, size_t nx, size_t ny, size_t nz,
    size_t direction, int p,
    const ivec3& start, const ivec3& end);
}


}
