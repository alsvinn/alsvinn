#pragma once

#include "alsfvm/types.hpp"
#include <cassert>

namespace alsfvm { namespace equation { namespace euler { 

	///
	/// The holder struct for all relevant variables for the euler flux
	/// These are supposed to be the conserved variables
	///
    template<int nsd>
    class ConservedVariables {
    public:
        typedef typename Types<nsd>::rvec rvec;
        typedef typename Types<nsd + 2>::rvec state_vector;
		__device__ __host__ ConservedVariables()
			:rho(0), m(0), E(0)
		{
			 // empty
		}
		__device__ __host__ ConservedVariables(real rho_, rvec m, real E_)
			: rho(rho_), m(m), E(E_)
		{
			// empty
		}

        __device__ __host__ ConservedVariables(const state_vector& in);
        

        __device__ __host__ real operator[](size_t index) const {
            assert(index < nsd + 2);
            return ((real*)this)[index];
        }

        __device__ __host__ real& operator[](size_t index) {
            assert(index < nsd + 2);
            return ((real*)this)[index];
        }

        __device__ __host__ static constexpr size_t size() {
            return 5;
        }

        __device__ __host__ bool operator==(const ConservedVariables& other) const {
            return rho == other.rho && m == other.m && E == other.E;
        }
		real rho;
		rvec m;
		real E;

    };

	///
	/// Computes the component difference
	/// \note Makes a new instance
	///
    template<int nsd>
	__device__ __host__ inline ConservedVariables<nsd> operator-(const ConservedVariables<nsd>& a, const ConservedVariables<nsd>& b) {
		return ConservedVariables<nsd>(a.rho - b.rho, a.m-b.m, a.E - b.E);
	}

	///
	/// Computes the component addition
	/// \note Makes a new instance
	///
    template<int nsd>
	__device__ __host__ inline ConservedVariables<nsd> operator+(const ConservedVariables<nsd>& a, const ConservedVariables<nsd>& b) {
		return ConservedVariables<nsd>(a.rho + b.rho, a.m + b.m, a.E + b.E);
	}

	///
	/// Computes the product of a and b (scalar times vector)
	/// \note Makes a new instance
	////
    template<int nsd>
	__device__ __host__ inline ConservedVariables<nsd> operator*(real a, const ConservedVariables<nsd>& b) {
		return ConservedVariables(a*b.rho, a*b.m, a*b.E);
	}

	///
	/// Computes the division of a by b 
	/// \note Makes a new instance
	///
    template<int nsd>
	__device__ __host__ inline ConservedVariables<nsd> operator/(const ConservedVariables<nsd>& a, real b) {
		return ConservedVariables<nsd>(a.rho / b, a.m / b, a.E / b);
	}

    template<>
    __device__ __host__ ConservedVariables<3>::ConservedVariables(const rvec5& in)
    : rho(in[0]), m(in[1], in[2], in[3]), E(in[4])
    {
        // empty
    }

    template<>
    __device__ __host__ ConservedVariables<2>::ConservedVariables(const rvec4& in)
        : rho(in[0]), m(in[1], in[2]), E(in[4])
    {
        // e
    }

    template<>
    __device__ __host__ ConservedVariables<1>::ConservedVariables(const rvec3& in)
        : rho(in[0]), m(in[1]), E(in[4])
    {
        // e
    }

} // namespace alsfvm

} // namespace numflux

} // namespace euler

