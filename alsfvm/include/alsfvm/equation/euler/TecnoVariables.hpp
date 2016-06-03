#pragma once
#include "alsfvm/types.hpp"

#include <array>
namespace alsfvm { namespace equation { namespace euler { 

    //!
    //! Simple class to hold the variables relevant for the Tecno
    //! scheme, see definition of \f$\vec{z}\f$ in
    //!
    //! http://www.cscamm.umd.edu/people/faculty/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
    //!
    //! eq (6.11). That is, we set
    //!
    //! \f[\vec{z} = \left(\begin{array}{l} \sqrt{\frac{\rho}{p}}\\ \sqrt{\frac{\rho}{p}}u\\ \sqrt{\frac{\rho}{p}}v\\ \sqrt{\rho p}\end{array}\right).\f]
    //!
    class TecnoVariables {
    public:
        TecnoVariables(real z1, real z2, real z3, real z4, real z5) :
            z(z1, z2, z3, z4, z5)
        {
            // empty
        }

        std::array<real, 5> z;
    };
} // namespace alsfvm
} // namespace equation
} // namespace euler
