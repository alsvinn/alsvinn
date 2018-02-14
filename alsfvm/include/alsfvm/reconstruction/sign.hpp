#pragma once
namespace alsfvm {
namespace reconstruction {
inline __device__  __host__  real sign(real a) {
    if (a > 0) {
        return 1;
    } else if (a == 0) {
        return 0;
    } else {
        return -1;
    }
}
}
}
