#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include <cassert>
#include <cmath>
namespace alsfvm {
namespace equation {
namespace euler {

template<int nsd>
class AllVariables : public ConservedVariables<nsd>,
    public ExtraVariables<nsd> {
public:

    typedef typename Types<nsd>::rvec rvec;
    __device__ __host__ AllVariables(real rho, rvec m, real E, real p, rvec u) :
        ConservedVariables<nsd>(rho, m, E), ExtraVariables<nsd>(p, u) {
    }

    __device__ __host__ const ConservedVariables<nsd>& conserved() const {
        return *this;
    }

    __device__ __host__ AllVariables(const ConservedVariables<nsd>& conserved,
        const ExtraVariables<nsd>& extra)
        : ConservedVariables<nsd>(conserved), ExtraVariables<nsd>(extra) {
    }

};




} // namespace alsfvm
} // namespace numflux
} // namespace euler

