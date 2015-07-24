#pragma once
#include "alsfvm/equation/CellComputer.hpp"

namespace alsfvm { namespace equation { 

    template<class Equation>
    class CPUCellComputer : public CellComputer {
    public:
        ///
        /// \brief computeExtraVariables computes the extra variables (eg. pressure for euler)
        /// \param[in] conservedVariables the conserved variables to read from
        /// \param[out] extraVariables the extra variables to write to
        ///
        virtual void computeExtraVariables(const volume::Volume& conservedVariables,
                                           volume::Volume& extraVariables);

    };
} // namespace alsfvm
} // namespace equation
