#pragma once
#include <vector>
namespace alsutils {

///
/// Small vector class to hold 3D data
///
template<class T>
struct vec1 {
    T x;

    __device__ __host__ vec1()
        : x(0) {

    }
    __device__ __host__ vec1(T x)
        : x(x) {
        // Empty
    }

    __device__ __host__ operator T() const {
        return x;
    }

    __device__ __host__ vec1(const vec1<T&>& other)
        : x(other.x) {

    }


    template<class S>
    __device__ __host__ vec1& operator=(const vec1<S>& other) {
        x = other.x;

        return *this;
    }

    __device__ __host__ bool operator==(const vec1& other) const {
        return other.x == x;
    }
#if __cplusplus > 199711L || WIN32
    ///
    /// Converts the vector to an std::vector<T>
    /// output is {x, y, z}
    ///
    std::vector<T> toStdVector() {
        return std::vector<T>({ x});
    }
#endif
    ///
    /// Converts the vector to the other type
    ///
    template<class S>
    __device__ __host__ vec1<S> convert() const {
        return vec1<S>(S(x));
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
    __device__ __host__ T dot(const vec1<T>& other) const {
        return x * other.x;
    }

    ///
    /// Returns 3 (number of components)
    ///
    __device__ __host__ static constexpr size_t size()  {
        return 1;
    }

    __device__ __host__ T norm() const {
        return abs(x);
    }

    template<class S>
    __device__ __host__ inline vec1<T>& operator+=( const vec1<S>& b) {
        x += b.x;

        return *this;
    }
};

///
/// Computes the component wise division of a by b.
/// Ie. the new vector will be
/// \f[(a_0/b_0, a_1/b_1, a_2/b_2)\f]
///
template<class T>
__device__ __host__ inline vec1<T> operator/(const vec1<T>& a,
    const vec1<T>& b) {
    return vec1<T>(a.x / b.x);
}

///
/// Computes the product \f$\mathrm{scalar} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec1<T> operator*(T scalar, const vec1<T>& a) {
    return vec1<T>(a.x * scalar);
}

///
/// Computes the difference \f$ \vec{a}-\vec{b}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec1<T> operator-(const vec1<T>& a,
    const vec1<T>& b) {
    return vec1<T>(a.x - b.x);
}

///
/// Computes the division \f$\frac{1}{\mathrm{scalar}} \vec{a}\f$
/// \note Creates a new vector instance
///
template<class T>
__device__ __host__ inline vec1<T> operator/(const vec1<T>& a, T scalar) {
    return vec1<T>(a.x / scalar);
}

///
/// Computes the product \f$\vec{a}+\vec{b}\f$
/// \note Creates a new vector instance.
///
template<class T, class S>
__device__ __host__ inline vec1<T> operator+(const vec1<T>& a,
    const vec1<S>& b) {
    return vec1<T>(a.x + b.x);
}
}
