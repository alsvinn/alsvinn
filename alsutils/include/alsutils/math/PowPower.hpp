#pragma once
#ifndef __CUDA_ARCH__
    #include <cmath>
#else
    #include <cuda.h>
#endif
namespace alsutils {
namespace math {
struct PowPower {

    __device__ __host__ static double power(double x, double p) {
#ifndef __CUDA_ARCH__ // commenting out this for cuda seemed to be necesary in a newer cuda version
        using namespace std;
#endif
#if 0
        double returnvalue = x;

        while (p > 1) {
            returnvalue *= x;
            p -= 1;
        }

        return returnvalue;
#else
        return pow(x, p);
#endif
    }
};
}
}
