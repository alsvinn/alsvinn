#pragma once
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace equation {
namespace burgers {

class PrimitiveVariables {
public:
    __device__ __host__ PrimitiveVariables()
        : u(0) {
        // empty
    }
    __device__ __host__ PrimitiveVariables(real u_)
        : u(u_) {
        // empty
    }

    real u;
};
} // namespace alsfvm
} // namespace equation
} // namespace burgers
