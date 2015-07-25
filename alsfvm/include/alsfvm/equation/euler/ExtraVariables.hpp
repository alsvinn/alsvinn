#pragma once
#include "alsfvm/types.hpp"


namespace alsfvm { namespace equation { namespace euler { 

	///
	/// The holder struct for all relevant variables for the euler flux
	/// These are supposed to be the extra variables (non-conserved)
	///
    class ExtraVariables {
    public:
		ExtraVariables(real p, real ux, real uy, real uz)
			: p(p), u(ux, uy, uz)
		{

		}

		ExtraVariables()
			: p(0), u(0, 0, 0)
		{

		}


		real p;
		rvec3 u;
    };

} // namespace alsfvm

} // namespace numflux

} // namespace euler

