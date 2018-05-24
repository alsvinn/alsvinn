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
#include "alsutils/vec1.hpp"
namespace alsutils {

///
/// Small vector class to hold 3D data
///
template<class T>
struct vec3 {
    T x;
    T y;
    T z;

    __device__ __host__ vec3()
        : x(0), y(0), z(0) {

    }

    __device__ __host__ vec3(T t) :
        x(t), y(t), z(t) {

    }
    __device__ __host__ vec3(T x, T y, T z)
        : x(x), y(y), z(z) {
        // Empty
    }

    __device__ __host__ vec3(T x, vec1<T> y, T z)
        : x(x), y(y.x), z(z) {
        // Empty
    }

    __device__ __host__ vec3(const vec3<T&>& other)
        : x(other.x), y(other.y), z(other.z) {

    }

    template<class S>
    __device__ __host__ vec3& operator=(const vec3<S>& other) {
        x = other.x;
        y = other.y;
        z = other.z;

        return *this;
    }

    __device__ __host__ bool operator==(const vec3& other) const {
        return other.x == x && other.y == y && other.z == z;
    }
#if __cplusplus > 199711L || WIN32
    ///
    /// Converts the vector to an std::vector<T>
    /// output is {x, y, z}
    ///
    std::vector<T> toStdVector() {
        return std::vector<T>({ x, y, z });
    }
#endif
    ///
    /// Converts the vector to the other type
    ///
    template<class S>
    __device__ __host__ vec3<S> convert() const  {
        return vec3<S>(S(x), S(y), S(z));
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
    __device__ __host__ T dot(const vec3<T>& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    ///
    /// Returns 3 (number of components)
    ///
    __device__ __host__ static constexpr size_t size()  {
        return 3;
    }

    template<class S>
    __device__ __host__ inline vec3<T>& operator+=( const vec3<S>& b) {
        x += b.x;
        y += b.y;
        z += b.z;

        return *this;
    }

    __host__ std::string str() const;

};

///
/// Computes the component wise division of a by b.
/// Ie. the new vector will be
/// \f[(a_0/b_0, a_1/b_1, a_2/b_2)\f]
///
template<class T>
__device__ __host__ inline vec3<T> operator/(const vec3<T>& a,
    const vec3<T>& b) {
    return vec3<T>(a.x / b.x, a.y / b.y, a.z / b.z);
}

///
/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec3<T> operator*(T scalar, const vec3<T>& a) {
    return vec3<T>(a.x * scalar, a.y * scalar, a.z * scalar);
}

///
/// Computes the product \f$\ (a_x\cdot b_x, a_y\cdot b_y, a_z\cdot b_z)\f$
/// \note Creates a new vector instance
///
template<class T, class S>
__device__ __host__ inline vec3<T> operator*(const vec3<T>& a,
    const vec3<S>& b) {
    return vec3<T>(a.x * b.x, a.y * b.y, a.z * b.z);
}

///
/// Computes the difference \f$ \vec{a}-\vec{b}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec3<T> operator-(const vec3<T>& a,
    const vec3<T>& b) {
    return vec3<T>(a.x - b.x, a.y - b.y, a.z - b.z);
}

///
/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec3<T> operator/(const vec3<T>& a, T scalar) {
    return vec3<T>(a.x / scalar, a.y / scalar, a.z / scalar);
}

///
/// Computes the product \f$\vec{a}+\vec{b}\f$
/// \note Creates a new vector instance.
///
template<class T, class S>
__device__ __host__ inline vec3<T> operator+(const vec3<T>& a,
    const vec3<S>& b) {
    return vec3<T>(a.x + b.x, a.y + b.y,  a.z + b.z);
}



}

template<typename T>
std::ostream& operator<<(std::ostream& os,
    const alsutils::vec3<T>& vec) {
    os << "[";

    for (int i = 0; i < 3; ++i) {
        os << vec[i];

        if (i < 2) {
            os << ", ";
        }
    }

    os << "]";
    return os;
}

template<class T>
inline __host__ std::string alsutils::vec3<T>::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

