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
#include <vector>
namespace alsutils {

///
/// Small vector class to hold 3D data
///
template<class T>
struct vec2 {
    T x;
    T y;

    __device__ __host__ vec2()
        : x(0), y(0) {

    }

    __device__ __host__ vec2(T t) :
        x(t), y(t) {

    }

    __device__ __host__ vec2(T x, T y)
        : x(x), y(y) {
        // Empty
    }



    __device__ __host__ vec2(const vec2<T&>& other)
        : x(other.x), y(other.y) {



    }

    template<class S>
    __device__ __host__ vec2& operator=(const vec2<S>& other) {
        x = other.x;
        y = other.y;


        return *this;
    }


    __device__ __host__ bool operator==(const vec2& other) const {
        return other.x == x && other.y == y;
    }
#if __cplusplus > 199711L || WIN32
    ///
    /// Converts the vector to an std::vector<T>
    /// output is {x, y}
    ///
    std::vector<T> toStdVector() {
        return std::vector<T>({ x, y });
    }
#endif
    ///
    /// Converts the vector to the other type
    ///
    template<class S>
    __device__ __host__ vec2<S> convert() const {
        return vec2<S>(S(x), S(y));
    }

    ///
    /// Returns the i-th component of the vector.
    ///
    __device__ __host__ const T& operator[](size_t i) const {
        // Note: We only store three numbers in this struct, hence this is safe
        return ((T*)this)[i];
    }

    ///
    /// Returns the i-th component of the vector.
    ///
    __device__ __host__ T& operator[](size_t i) {
        // Note: We only store three numbers in this struct, hence this is safe
        return ((T*)this)[i];
    }

    ///
    /// Computes the dot (scalar) product
    ///
    __device__ __host__ T dot(const vec2<T>& other) const {
        return x * other.x + y * other.y;
    }

    ///
    /// Returns 3 (number of components)
    ///
    __device__ __host__ static constexpr size_t size() {
        return 2;
    }

    template<class S>
    __device__ __host__ inline vec2<T>& operator+=( const vec2<S>& b) {
        x += b.x;
        y += b.y;

        return *this;
    }
};

///
/// Computes the component wise division of a by b.
/// Ie. the new vector will be
/// \f[(a_0/b_0, a_1/b_1, a_2/b_2)\f]
///
template<class T>
__device__ __host__ inline vec2<T> operator/(const vec2<T>& a,
    const vec2<T>& b) {
    return vec2<T>(a.x / b.x, a.y / b.y);
}

///
/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec2<T> operator*(T scalar, const vec2<T>& a) {
    return vec2<T>(a.x * scalar, a.y * scalar);
}

///
/// Computes the difference \f$ \vec{a}-\vec{b}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec2<T> operator-(const vec2<T>& a,
    const vec2<T>& b) {
    return vec2<T>(a.x - b.x, a.y - b.y);
}

///
/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec2<T> operator/(const vec2<T>& a, T scalar) {
    return vec2<T>(a.x / scalar, a.y / scalar);
}

///
/// Computes the product \f$\vec{a}+\vec{b}\f$
/// \note Creates a new vector instance.
///
template<class T, class S>
__device__ __host__ inline vec2<T> operator+(const vec2<T>& a,
    const vec2<S>& b) {
    return vec2<T>(a.x + b.x, a.y + b.y);
}

}
