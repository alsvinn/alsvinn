#pragma once
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace equation {
namespace buckleyleverett {

///
/// The holder struct for all relevant variables for the Burgers flux
/// These are supposed to be the conserved variables
///
class ConservedVariables {
public:
    __device__ __host__ ConservedVariables()
        : u(0) {
        // empty
    }
    __device__ __host__ ConservedVariables(real u_)
        : u(u_) {
        // empty
    }

    __device__ __host__ ConservedVariables(const rvec1& u_)
        : u(u_.x) {
        // empty
    }

    __device__ __host__ static constexpr size_t size() {
        return 1;
    }

    __device__ __host__ real& operator[](size_t index) {
        return u;
    }

    __device__ __host__ real operator[](size_t index) const {
        return u;
    }

    real u;
};

///
/// Computes the component difference
/// \note Makes a new instance
///
__device__ __host__ inline ConservedVariables operator-(const
    ConservedVariables& a, const ConservedVariables& b) {
    return ConservedVariables(a.u - b.u);
}

///
/// Computes the component addition
/// \note Makes a new instance
///
__device__ __host__ inline ConservedVariables operator+(const
    ConservedVariables& a, const ConservedVariables& b) {
    return ConservedVariables(a.u + b.u);
}

///
/// Computes the product of a and b (scalar times vector)
/// \note Makes a new instance
////
__device__ __host__ inline ConservedVariables operator*(real a,
    const ConservedVariables& b) {
    return ConservedVariables(a * b.u);
}

///
/// Computes the division of a by b
/// \note Makes a new instance
////
__device__ __host__ inline ConservedVariables operator/(const
    ConservedVariables& a, real b) {
    return ConservedVariables(a.u / b);
}



} // namespace alsfvm
} // namespace equation
} // namespace buckleyleverett
