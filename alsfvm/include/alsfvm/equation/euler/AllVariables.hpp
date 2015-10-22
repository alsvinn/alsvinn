#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
#include <cassert>
#include <cmath>
namespace alsfvm { namespace equation { namespace euler { 
    class AllVariables : public ConservedVariables, public ExtraVariables {
    public:
		__device__ __host__ AllVariables(real rho, real mx, real my, real mz, real E, real p, real ux, real uy, real uz) :
			ConservedVariables(rho, mx, my, mz, E), ExtraVariables(p, ux, uy, uz)
		{
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
} // namespace numflux
} // namespace euler

