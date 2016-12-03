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

