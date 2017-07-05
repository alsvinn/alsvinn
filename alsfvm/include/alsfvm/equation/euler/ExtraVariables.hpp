#pragma once
#include "alsfvm/types.hpp"


namespace alsfvm { namespace equation { namespace euler { 

	///
	/// The holder struct for all relevant variables for the euler flux
	/// These are supposed to be the extra variables (non-conserved)
	///
    template<int nsd>
    class ExtraVariables {
    public:

        typedef typename Types<nsd>::rvec rvec;


		__device__ __host__ ExtraVariables(real p, rvec u)
			: p(p), u(u)
		{

		}

        template<class T>
        __device__ __host__ ExtraVariables(T p, T ux, T uy, T uz)
            : p(p), u(rvec3{ux, uy, uz})
        {
            static_assert(nsd==3 ||sizeof(T)==0, "Only for 3 dimensions!");
        }

        template<class T>
        __device__ __host__ ExtraVariables(T p, T ux, T uy)
            : p(p), u(rvec2{ux, uy})
        {
            static_assert(nsd==2 ||sizeof(T)==0, "Only for 3 dimensions!");
        }

		__device__ __host__ ExtraVariables()
			: p(0), u(0)
		{

		}


		real p;
		rvec u;
    };


} // namespace alsfvm

} // namespace numflux

} // namespace euler

