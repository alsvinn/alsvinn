#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/euler/ConservedVariables.hpp"
#include "alsfvm/equation/euler/ExtraVariables.hpp"
namespace alsfvm { namespace equation { namespace euler { 
    class AllVariables : public ConservedVariables, public ExtraVariables {
    public:
		AllVariables(real rho, real mx, real my, real mz, real E, real p, real ux, real uy, real uz) :
			ConservedVariables(rho, mx, my, mz, E), ExtraVariables(p, ux, uy, uz)
		{

		}

		const ConservedVariables& conserved() const {
			return *this;
		}
		
    };

} // namespace alsfvm
} // namespace numflux
} // namespace euler

