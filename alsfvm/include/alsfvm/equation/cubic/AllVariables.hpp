#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/cubic/ConservedVariables.hpp"
#include "alsfvm/equation/cubic/ExtraVariables.hpp"
namespace alsfvm { namespace equation { namespace cubic { 

class AllVariables : public ConservedVariables, public ExtraVariables {
public:
    __device__ __host__ AllVariables(real u)
      :  ConservedVariables(u)
    {
        // empty
    }

    __device__ __host__ const ConservedVariables& conserved() const {
        return *this;
    }

    __device__ __host__ AllVariables(const ConservedVariables& conserved, const ExtraVariables& extra)
        : ConservedVariables(conserved), ExtraVariables(extra)
    {
    }
};

} // namespace alsfvm
} // namespace equation
} // namespace cubic