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

///
/// Computes the maximum wavespeed across all direction
/// \param conservedVariables the conserved variables (density, momentum, Energy for Euler)
/// \param extraVariables the extra variables (pressure and velocity for Euler)
/// \return the maximum wave speed (absolute value)
///
template<class Equation>
real CPUCellComputer<Equation>::computeMaxWaveSpeed(const volume::Volume& conservedVariables,
	const volume::Volume& extraVariables) {
	real maxWaveSpeed = 0;

	volume::for_each_cell<typename Equation::ConservedVariables,
		typename Equation::ExtraVariables>(conservedVariables, extraVariables, [&maxWaveSpeed](const euler::ConservedVariables& conserved,
		const euler::ExtraVariables& extra, size_t index) {

		const real waveSpeedX = Equation::template computeWaveSpeed<0>(conserved, extra);
		maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedX);

		const real waveSpeedY = Equation::template computeWaveSpeed<1>(conserved, extra);
		maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedY);

		const real waveSpeedZ = Equation::template computeWaveSpeed<1>(conserved, extra);
		maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedZ);
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

	volume::for_each_cell<typename Equation::ConservedVariables,
        typename Equation::ExtraVariables>(conservedVariables, extraVariables, [&obeys](const typename Equation::ConservedVariables& conserved,
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
		if (!Equation::obeysConstraints(conserved, extra)) {
			obeys = false;
		}
	});

	return obeys;
}

template class CPUCellComputer<euler::Euler>;
}
}
