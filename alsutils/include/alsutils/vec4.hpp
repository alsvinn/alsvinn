#pragma once
#include <vector>
#include "alsutils/vec2.hpp"
namespace alsutils {

///
/// Small vector class to hold 3D data
///
template<class T>
struct vec4 {
    T x;
    T y;
    T z;
    T v;

    __device__ __host__ vec4()
        : x(0), y(0), z(0), v(0) {

    }
    __device__ __host__ vec4(T x, T y, T z, T v)
        : x(x), y(y), z(z), v(v) {
        // Empty
    }

    __device__ __host__ vec4(T t) :
        x(t), y(t), z(t), v(t) {

    }

    __device__ __host__ vec4(T x, vec2<T> y, T z)
        : x(x), y(y.x), z(y.y), v(z) {
        // Empty
    }


    __device__ __host__ vec4(const vec4<T&>& other)
        : x(other.x), y(other.y), z(other.z), v(other.v) {

    }

    template<class S>
    __device__ __host__ vec4& operator=(const vec4<S>& other) {
        x = other.x;
        y = other.y;
        z = other.z;
        v = other.v;


        return *this;
    }
    __device__ __host__ bool operator==(const vec4& other) const {
        return other.x == x && other.y == y && other.z == z && other.v == v;
    }
#if __cplusplus > 199711L || WIN32
    ///
    /// Converts the vector to an std::vector<T>
    /// output is {x, y, z, v, w}
    ///
    std::vector<T> toStdVector() {
        return std::vector<T>({ x, y, z, v });
    }
#endif
    ///
    /// Converts the vector to the other type
    ///
    template<class S>
    __device__ __host__ vec4<S> convert() {
        return vec4<S>(S(x), S(y), S(z), S(v));
    }

    ///
    /// Returns the i-th component of the vector.
    ///
    __device__ __host__ const T& operator[](size_t i) const {
        // Note: We only store five numbers in this struct, hence this is safe
        return ((T*)this)[i];
    }

    ///
    /// Returns the i-th component of the vector.
    ///
    __device__ __host__ T& operator[](size_t i) {
        // Note: We only store five numbers in this struct, hence this is safe
        return ((T*)this)[i];
    }

    ///
    /// Computes the dot (scalar) product
    ///
    __device__ __host__ T dot(const vec4<T>& other) const {
        return x * other.x + y * other.y + z * other.z + v * other.v;
    }

    ///
    /// Returns 5 (number of components)
    ///
    __device__ __host__ static constexpr size_t size() {
        return 4;
    }

    template<class S>
    __device__ __host__ inline vec4<T>& operator+=( const vec4<S>& b) {
        x += b.x;
        y += b.y;
        z += b.z;
        v += b.v;

        return *this;
    }
};

///
/// Computes the component wise division of a by b.
/// Ie. the new vector will be
/// \f[(a_0/b_0, a_1/b_1, a_2/b_2)\f]
///
template<class T>
__device__ __host__ inline vec4<T> operator/(const vec4<T>& a,
    const vec4<T>& b) {
    return vec4<T>(a.x / b.x, a.y / b.y, a.z / b.z, a.v / b.v);
}

///
/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec4<T> operator*(T scalar, const vec4<T>& a) {
    return vec4<T>(a.x * scalar, a.y * scalar, a.z * scalar, a.v * scalar);
}

///
/// Computes the difference \f$ \vec{a}-\vec{b}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec4<T> operator-(const vec4<T>& a,
    const vec4<T>& b) {
    return vec4<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.v - b.v);
}

///
/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec4<T> operator/(const vec4<T>& a, T scalar) {
    return vec4<T>(a.x / scalar, a.y / scalar, a.z / scalar, a.v / scalar);
}

///
/// Computes the product \f$\vec{a}+\vec{b}\f$
/// \note Creates a new vector instance.
///
template<class T, class S>
__device__ __host__ inline vec4<T> operator+(const vec4<T>& a,
    const vec4<S>& b) {
    return vec4<T>(a.x + b.x, a.y + b.y, a.z + b.z.a.v + b.v);
}




}
template<typename T>
inline std::ostream& operator<<(std::ostream& os,
    const alsutils::vec4<T>& vec) {
    os << "[";

    for (int i = 0; i < 4; ++i) {
        os << vec[i];

        if (i < 3) {
            os << ", ";
        }
    }

    os << "]";
    return os;
}
