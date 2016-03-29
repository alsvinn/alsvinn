#include "alsfvm/equation/CPUCellComputer.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm { namespace equation {

template<class Equation>
CPUCellComputer<Equation>::CPUCellComputer(simulator::SimulatorParameters &parameters)
    : parameters(static_cast<typename Equation::Parameters&>(parameters.getEquationParameters()))
{

}

template<class Equation>
void CPUCellComputer<Equation>::computeExtraVariables(const volume::Volume &conservedVariables,
                                                      volume::Volume &extraVariables)
{
    Equation eq(parameters);
    volume::transform_volume<typename Equation::ConservedVariables,
            typename Equation::ExtraVariables>(
                conservedVariables, extraVariables,
                             [&](const typename Equation::ConservedVariables& in)
                                -> typename Equation::ExtraVariables
    {
        return eq.computeExtra(in);
    });
}


template<class Equation>
real CPUCellComputer<Equation>::computeMaxWaveSpeed(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables, size_t direction) {
    Equation eq(parameters);
	real maxWaveSpeed = 0;
    assert(direction < 3);
	volume::for_each_cell<typename Equation::ConservedVariables,
        typename Equation::ExtraVariables>(conservedVariables, extraVariables, [&maxWaveSpeed, direction,&eq](const euler::ConservedVariables& conserved,
		const euler::ExtraVariables& extra, size_t index) {
        if (direction == 0) {
            const real waveSpeedX = eq.template computeWaveSpeed<0>(conserved, extra);
            maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedX);
        } else if(direction == 1) {
            const real waveSpeedY = eq.template computeWaveSpeed<1>(conserved, extra);
            maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedY);
        }    else if(direction == 2) {
            const real waveSpeedZ = eq.template computeWaveSpeed<2>(conserved, extra);
            maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedZ);
        }
    });

	return maxWaveSpeed;
}

/// 
/// Checks if all the constraints for the equation are met
///	\param conservedVariables the conserved variables (density, momentum, Energy for Euler)
/// \param extraVariables the extra variables (pressure and velocity for Euler)
/// \return true if it obeys the constraints, false otherwise
/// \todo Tidy up the way we check for nan and inf
///
template<class Equation>
bool CPUCellComputer<Equation>::obeysConstraints(const volume::Volume& conservedVariables,
	const volume::Volume& extraVariables) {

	bool obeys = true;
    Equation eq(parameters);
	volume::for_each_cell<typename Equation::ConservedVariables,
        typename Equation::ExtraVariables>(conservedVariables, extraVariables, [&obeys,&eq](const typename Equation::ConservedVariables& conserved,
        const typename Equation::ExtraVariables& extra, size_t index) {
        // Check for nan and inf:
        const real* conservedAsRealPtr = (const real*)&conserved;

        for (size_t i = 0; i < sizeof(conserved)/sizeof(real); i++) {
            if (std::isnan(conservedAsRealPtr[i]) || std::isinf(conservedAsRealPtr[i])) {
                obeys = false;
            }
        }


        const real* extraAsRealPtr = (const real*)&extra;

        for (size_t i = 0; i < sizeof(extra)/sizeof(real); i++) {
            if (std::isnan(extraAsRealPtr[i]) || std::isinf(extraAsRealPtr[i])) {
                obeys = false;
            }
        }
        if (!eq.obeysConstraints(conserved, extra)) {
			obeys = false;
		}
	});

    return obeys;
}

template<class Equation>
void CPUCellComputer<Equation>::computeFromPrimitive(const volume::Volume &primitiveVariables,
                                                     volume::Volume &conservedVariables,
                                                     volume::Volume &extraVariables)
{
   Equation eq(parameters);
    volume::transform_volume<typename Equation::PrimitiveVariables,
            typename Equation::ExtraVariables>(
                primitiveVariables, extraVariables,
                             [&](const typename Equation::PrimitiveVariables& in)
                                -> typename Equation::ExtraVariables
    {
        return eq.computeExtra(in);
    });


    volume::transform_volume<typename Equation::PrimitiveVariables,
            typename Equation::ConservedVariables>(
                primitiveVariables, conservedVariables,
                             [&](const typename Equation::PrimitiveVariables& in)
                                -> typename Equation::ConservedVariables
    {
        return eq.computeConserved(in);
    });
}

ALSFVM_EQUATION_INSTANTIATE(CPUCellComputer);
}
}
