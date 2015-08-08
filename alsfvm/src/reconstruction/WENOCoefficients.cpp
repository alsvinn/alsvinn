#include "alsfvm/reconstruction/WENOCoefficients.hpp"

namespace alsfvm { namespace reconstruction { 
	template<>
	real WENOCoefficients<2>::coefficients[] = {
		2.0 / 3.0, 1.0 / 3.0
	};


	template<>
	real WENOCoefficients<3>::coefficients[] = {
		3.0 / 10.0, 3.0 / 5.0, 1.0 / 10.0
	};
}
}
