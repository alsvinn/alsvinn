#pragma once
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace reconstruction {

template<int k>
class ENOCoeffiecients {
    public:
        ///
        /// \brief coefficients are the ENO coefficients.
        ///
        /// coefficients[r][i] indexes the coefficient with shift r and term i
        ///
        /// See http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19980007543.pdf
        /// for description.
        ///
        static real coefficients[][k];
};
}
}
