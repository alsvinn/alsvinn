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


    ///
    /// Computes the component difference
    /// \note Makes a new instance
    ///
	__device__ __host__ inline AllVariables operator-(const AllVariables& a, const AllVariables& b) {
        assert(false);
    }

    ///
    /// Computes the component addition
    /// \note Makes a new instance
    ///
	__device__ __host__ inline AllVariables operator+(const AllVariables& a, const AllVariables& b) {
        assert(false);
    }

    ///
    /// Computes the product of a and b (scalar times vector)
    /// \note Makes a new instance
    ////
	__device__ __host__ inline AllVariables operator*(real a, const AllVariables& b) {
        assert(false);
    }

    ///
    /// Computes the division of a by b
    /// \note Makes a new instance
    ////
	__device__ __host__ inline AllVariables operator/(const AllVariables& a, real b) {
        assert(false);
    }

} // namespace alsfvm
} // namespace numflux
} // namespace euler

