#pragma once
#include "alsfvm/integrator/Integrator.hpp"

namespace alsfvm { namespace integrator { 

	///
	/// Does 2nd order RungeKutta-integrator. In other words,
	/// this solves the system
	///   \f[U_t=F(U)\f]
	///
	/// by setting 
	///   \f[U_0 = U(0)\f]
	///
	/// and then for each \f$n>0\f$, we set
	///
	/// 
    class RungeKutta2 : public Integrator {
    public:

    };
} // namespace alsfvm
} // namespace integrator
