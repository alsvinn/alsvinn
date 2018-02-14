#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/EquationParameters.hpp"

namespace alsfvm {
namespace equation {
namespace euler {

class EulerParameters : public EquationParameters {
public:
    __device__ __host__ EulerParameters()
        : gamma(1.4) {
        // empty
    }

    __device__ __host__ void setGamma(real gamma_) {
        gamma = gamma_;
    }

    __device__ __host__ real getGamma() const {
        return gamma;
    }

private:
    real gamma;

};
} // namespace euler
} // namespace equation
} // namespace alsfvm
