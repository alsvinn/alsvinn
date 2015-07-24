#include "alsfvm/equation/CPUCellComputer.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
namespace alsfvm { namespace equation {

template<class Equation>
void CPUCellComputer<Equation>::computeExtraVariables(const volume::Volume &conservedVariables,
                                                      volume::Volume &extraVariables)
{
    volume::transform_volume<typename Equation::ConservedVariables,
            typename Equation::ExtraVariables>(
                conservedVariables, extraVariables,
                             [](const typename Equation::ConservedVariables& in)
                                -> typename Equation::ExtraVariables
    {
        return Equation::computeExtra(in);
    });
}

template class CPUCellComputer<euler::Euler>;
}
}
