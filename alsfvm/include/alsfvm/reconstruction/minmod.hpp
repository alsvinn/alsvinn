#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/sign.hpp"

namespace alsfvm {
namespace reconstruction {
inline __device__  __host__ real minmod(real a, real b) {
    if (sign(a) == sign(b)) {
        return sign(a) * fmin(fabs(a), fabs(b));
    } else {
        return 0;
    }
}

inline __device__  __host__  real minmod(real a, real b, real c) {
    if (sign(a) == sign(b) && sign(b) == sign(c)) {
        return sign(a) * fmin(fabs(a), fmin(fabs(b), fabs(c)));
    } else {
        return 0;
    }
}
}
}
