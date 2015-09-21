#pragma once

#include "alsfvm/types.hpp"

namespace alsfvm { namespace equation { namespace euler { 

	///
	/// The holder struct for all relevant variables for the euler flux
	/// These are supposed to be the conserved variables
	///
    class ConservedVariables {
    public:
		__device__ __host__ ConservedVariables()
			:rho(0), m(0, 0, 0), E(0)
		{
			 // empty
		}
		__device__ __host__ ConservedVariables(real rho, real mx, real my, real mz, real E)
			: rho(rho), m(mx, my, mz), E(E)
		{
			// empty
		}

		real rho;
		rvec3 m;
		real E;

    };

	///
	/// Computes the component difference
	/// \note Makes a new instance
	///
	__device__ __host__ inline ConservedVariables operator-(const ConservedVariables& a, const ConservedVariables& b) {
		return ConservedVariables(a.rho - b.rho, a.m.x - b.m.x, a.m.y - b.m.y, a.m.z - b.m.z, a.E - b.E);
	}

	///
	/// Computes the component addition
	/// \note Makes a new instance
	///
	__device__ __host__ inline ConservedVariables operator+(const ConservedVariables& a, const ConservedVariables& b) {
		return ConservedVariables(a.rho + b.rho, a.m.x + b.m.x, a.m.y + b.m.y, a.m.z + b.m.z, a.E + b.E);
	}

	///
	/// Computes the product of a and b (scalar times vector)
	/// \note Makes a new instance
	////
	__device__ __host__ inline ConservedVariables operator*(real a, const ConservedVariables& b) {
		return ConservedVariables(a*b.rho, a*b.m.x, a*b.m.y, a*b.m.z, a*b.E);
	}

	///
	/// Computes the division of a by b 
	/// \note Makes a new instance
	////
	__device__ __host__ inline ConservedVariables operator/(const ConservedVariables& a, real b) {
		return ConservedVariables(a.rho / b, a.m.x / b, a.m.y / b, a.m.z / b, a.E / b);
	}

} // namespace alsfvm

} // namespace numflux

} // namespace euler

