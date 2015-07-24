#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm { namespace equation { 

///
/// \brief The CellComputer class defines some useful per cell computations
///
    class CellComputer {
    public:
        ///
        /// \brief computeExtraVariables computes the extra variables (eg. pressure for euler)
        /// \param[in] conservedVariables the conserved variables to read from
        /// \param[out] extraVariables the extra variables to write to
        ///
        virtual void computeExtraVariables(const volume::Volume& conservedVariables,
                                           volume::Volume& extraVariables) = 0;



    };
} // namespace alsfvm
} // namespace equation
