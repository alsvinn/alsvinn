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

        inline __device__ __host__ ConservedVariables()
			:rho(0), m(0), E(0)
		{
			 // empty
		}



        template<class ValueType>
        inline __device__ __host__ ConservedVariables(real rho_,

                                               const typename Types<nsd>::template vec<ValueType>& m_, real E_)
            : rho(rho_), m(m_.template convert<real>()), E(E_)
		{
			// empty
		}


        template<class T>
        inline __device__ __host__ ConservedVariables(T rho_, T mx, T my, T mz, T E)
            : rho(rho_), m(rvec3{mx, my, mz}), E(E)
        {
            static_assert(nsd==3 ||sizeof(T)==0, "Only for 3 dimensions!");
        }

        template<class T>
        inline __device__ __host__ ConservedVariables(T rho_, T mx, T my, T E)
            : rho(rho_), m(rvec2{mx, my}), E(E)
        {
            static_assert(nsd==2 ||sizeof(T)==0, "Only for 3 dimensions!");
        }

        template<class T>
        inline __device__ __host__ ConservedVariables(T rho_, T mx, T E)
            : rho(rho_), m(rvec1{mx}), E(E)
        {
            static_assert(nsd==1 ||sizeof(T)==0, "Only for 3 dimensions!");
        }



        __device__ __host__ ConservedVariables(const state_vector& in);
        

        inline __device__ __host__ real operator[](size_t index) const {
            assert(index < nsd + 2);
            return ((real*)this)[index];
        }

        inline __device__ __host__ real& operator[](size_t index) {
            assert(index < nsd + 2);
            return ((real*)this)[index];
        }

        inline __device__ __host__ static constexpr size_t size() {
            return nsd + 2;
        }

        inline __device__ __host__ bool operator==(const ConservedVariables& other) const {
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
        return ConservedVariables<nsd>(a*b.rho, a*b.m, a*b.E);
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
    __device__ __host__ inline ConservedVariables<3>::ConservedVariables(const rvec5& in)
    : rho(in[0]), m(in[1], in[2], in[3]), E(in[4])
    {
        // empty
    }

    template<>
    __device__ __host__ inline ConservedVariables<2>::ConservedVariables(const rvec4& in)
        : rho(in[0]), m(in[1], in[2]), E(in[3])
    {
        // e
    }

    template<>
    __device__ __host__ inline ConservedVariables<1>::ConservedVariables(const rvec3& in)
        : rho(in[0]), m(in[1]), E(in[2])
    {
        // e
    }

} // namespace alsfvm

} // namespace numflux

} // namespace euler

