#pragma once
#include <vector>
#include "alsutils/vec3.hpp"
namespace alsutils {

///
/// Small vector class to hold 3D data
///
template<class T>
struct vec5 {
    T x;
    T y;
    T z;
    T v;
    T w;


    __device__ __host__ vec5(T t) :
        x(t), y(t), z(t), v(t), w(t) {

    }
    __device__ __host__ vec5()
        : x(0), y(0), z(0), v(0), w(0) {

    }
    __device__ __host__ vec5(T x, T y, T z, T v, T w)
        : x(x), y(y), z(z), v(v), w(w) {
        // Empty
    }

    __device__ __host__ vec5(T x, vec3<T> y, T z)
        : x(x), y(y.x), z(y.y), v(y.z), w(z) {
        // Empty
    }

    __device__ __host__ vec5(const vec5<T&>& other)
        : x(other.x), y(other.y), z(other.z), v(other.v), w(other.w) {

    }


    template<class S>
    __device__ __host__ vec5& operator=(const vec5<S>& other) {
        x = other.x;
        y = other.y;
        z = other.z;
        v = other.v;
        w = other.w;

        return *this;
    }

    __device__ __host__ bool operator==(const vec5& other) const {
        return other.x == x && other.y == y && other.z == z && other.v == v
            && other.w == w;
    }
#if __cplusplus > 199711L || WIN32
    ///
    /// Converts the vector to an std::vector<T>
    /// output is {x, y, z, v, w}
    ///
    std::vector<T> toStdVector() {
        return std::vector<T>({ x, y, z, v, w });
    }
#endif
    ///
    /// Converts the vector to the other type
    ///
    template<class S>
    __device__ __host__ vec5<S> convert() {
        return vec5<S>(S(x), S(y), S(z), S(v), S(w));
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
    __device__ __host__ T dot(const vec5<T>& other) const {
        return x * other.x + y * other.y + z * other.z + v * other.v + w * other.w;
    }

    ///
    /// Returns 5 (number of components)
    ///
    __device__ __host__ static constexpr size_t size() {
        return 5;
    }

    template<class S>
    __device__ __host__ inline vec5<T>& operator+=( const vec5<S>& b) {
        x += b.x;
        y += b.y;
        z += b.z;
        v += b.v;
        w += b.w;
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
__device__ __host__ inline vec5<T> operator/(const vec5<T>& a,
    const vec5<T>& b) {
    return vec5<T>(a.x / b.x, a.y / b.y, a.z / b.z, a.v / b.v, a.w / b.w);
}

///
/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec5<T> operator*(T scalar, const vec5<T>& a) {
    return vec5<T>(a.x * scalar, a.y * scalar, a.z * scalar, a.v * scalar,
            a.w * scalar);
}

///
/// Computes the difference \f$ \vec{a}-\vec{b}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec5<T> operator-(const vec5<T>& a,
    const vec5<T>& b) {
    return vec5<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.v - b.v, a.w - b.w);
}

///
/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec5<T> operator/(const vec5<T>& a, T scalar) {
    return vec5<T>(a.x / scalar, a.y / scalar, a.z / scalar, a.v / scalar,
            a.w / scalar);
}

///
/// Computes the product \f$\vec{a}+\vec{b}\f$
/// \note Creates a new vector instance.
///
template<class T, class S>
__device__ __host__ inline vec5<T> operator+(const vec5<T>& a,
    const vec5<S>& b) {
    return vec5<T>(a.x + b.x, a.y + b.y, a.z + b.z, a.v + b.v, a.w + b.w);
}





}


template<typename T>
std::ostream& operator<<(std::ostream& os,
    const alsutils::vec5<T>& vec) {
    os << "[";

    for (int i = 0; i < 5; ++i) {
        os << vec[i];

        if (i < 4) {
            os << ", ";
        }
    }

    os << "]";
    return os;
}

template<class T>
inline __host__ std::string alsutils::vec5<T>::str() const {
    std::stringstream ss;
    ss << *this;
    return ss.str();
}

