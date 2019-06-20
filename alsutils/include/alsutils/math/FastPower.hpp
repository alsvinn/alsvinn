#pragma once

namespace alsutils {
namespace math {

// This will be nicer when we can finally upgrade to C++14
template<int p>
struct FastPower {
    __device__ __host__  static double power(double x, double) {
        return power_internal(x);
    }

    __device__ __host__ static double power_internal(double x);
};

template<int p>
__device__ __host__  double FastPower<p>::power_internal(double x){
    return x*FastPower<p-1>::power_internal(x);
}

template<>
inline __device__ __host__  double FastPower<1>::power_internal(double x) {
    return x;
}
}
}
