#pragma once
#include <vector>
namespace alsutils {

///
/// Small vector class to hold 3D data
///
template<class T>
struct vec6 {
    T x;
    T y;
    T z;
    T v;
    T w;
    T u;

    __device__ __host__ vec6()
        : x(0), y(0), z(0), v(0), w(0), u(0) {

    }
    __device__ __host__ vec6(T x, T y, T z, T v, T w, T u)
        : x(x), y(y), z(z), v(v), w(w), u(0) {
        // Empty
    }

    __device__ __host__ bool operator==(const vec6& other) const {
        return other.x == x && other.y == y && other.z == z && other.v == v
            && other.w == w && other.u == u;
    }
#if __cplusplus > 199711L || WIN32
    ///
    /// Converts the vector to an std::vector<T>
    /// output is {x, y, z, v, w}
    ///
    std::vector<T> toStdVector() {
        return std::vector<T>({ x, y, z, v, w, u });
    }
#endif
    ///
    /// Converts the vector to the other type
    ///
    template<class S>
    __device__ __host__ vec6<S> convert() {
        return vec6<S>(S(x), S(y), S(z), S(v), S(w), S(u));
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
    __device__ __host__ T dot(const vec6<T>& other) const {
        return x * other.x + y * other.y + z * other.z + v * other.v + w * other.w + u *
            other.u;
    }

    ///
    /// Returns 5 (number of components)
    ///
    __device__ __host__ static constexpr size_t size() {
        return 6;
    }
};

///
/// Computes the component wise division of a by b.
/// Ie. the new vector will be
/// \f[(a_0/b_0, a_1/b_1, a_2/b_2)\f]
///
template<class T>
__device__ __host__ inline vec6<T> operator/(const vec6<T>& a,
    const vec6<T>& b) {
    return vec6<T>(a.x / b.x, a.y / b.y, a.z / b.z, a.v / b.v, a.w / b.w,
            a.u / b.u);
}

///
/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec6<T> operator*(T scalar, const vec6<T>& a) {
    return vec6<T>(a.x * scalar, a.y * scalar, a.z * scalar, a.v * scalar,
            a.w * scalar, a.u * scalar);
}

///
/// Computes the difference \f$ \vec{a}-\vec{b}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec6<T> operator-(const vec6<T>& a,
    const vec6<T>& b) {
    return vec6<T>(a.x - b.x, a.y - b.y, a.z - b.z, a.v - b.v, a.w - b.w,
            a.u - b.u);
}

///
/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec6<T> operator/(const vec6<T>& a, T scalar) {
    return vec6<T>(a.x / scalar, a.y / scalar, a.z / scalar, a.v / scalar,
            a.w / scalar, a.u / scalar);
}

///
/// Computes the product \f$\vec{a}+\vec{b}\f$
/// \note Creates a new vector instance.
///
template<class T>
__device__ __host__ inline vec6<T> operator+(const vec6<T>& a,
    const vec6<T>& b) {
    return vec6<T>(a.x + b.x, a.y + b.y, a.z + b.z.a.v + b.v, a.w + b.w, a.u + b.u);
}


}

template<typename T>
inline std::ostream& operator<<(std::ostream& os,
    const alsutils::vec6<T>& vec) {
    os << "[";

    for (int i = 0; i < 6; ++i) {
        os << vec[i];

        if (i < 5) {
            os << ", ";
        }
    }

    os << "]";
    return os;
}
