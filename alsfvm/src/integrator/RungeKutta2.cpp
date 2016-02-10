#include "alsfvm/integrator/RungeKutta2.hpp"
#include <iostream>
namespace alsfvm { namespace integrator { 

///
/// Returns the number of substeps this integrator uses.
/// Since this is second order RK, we need two subtimesteps
///
/// \returns 2
///
RungeKutta2::RungeKutta2(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux)
    : numericalFlux(numericalFlux)
{

}

size_t RungeKutta2::getNumberOfSubsteps() const {
    return 2;
}


real RungeKutta2::performSubstep(const std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
                                 rvec3 spatialCellSizes, real dt, real cfl,
                                 volume::Volume& output, size_t substep,
                                 const simulator::TimestepInformation& timestepInformation) {
    // We compute U + dt * F(U)


    // Compute F(U)
	rvec3 waveSpeeds(0, 0, 0);
    numericalFlux->computeFlux(*inputConserved[substep], waveSpeeds, true,  output);

	if (substep == 0) {
        dt = computeTimestep(waveSpeeds, spatialCellSizes, cfl, timestepInformation);
	}

	rvec3 cellScaling(dt / spatialCellSizes.x,
		dt / spatialCellSizes.y,
		dt / spatialCellSizes.z);

    output *= cellScaling.x;
    output += *inputConserved[substep];

    if (substep == 1) {
        // 0.5 * (U+U')
        output += *inputConserved[0];
        output *= 0.5;
    }

	return dt;
}
}
}
