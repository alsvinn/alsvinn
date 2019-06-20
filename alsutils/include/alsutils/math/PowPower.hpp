#pragma once
#include <cmath>

namespace alsutils {
namespace math {
struct PowPower {

    __device__ __host__ static double power(double x, double p) {
        using namespace std;
        return pow(x, p);
    }
};
}
}
