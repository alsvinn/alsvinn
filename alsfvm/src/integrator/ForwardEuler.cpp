#include "alsfvm/integrator/ForwardEuler.hpp"
#include <iostream>
namespace alsfvm { namespace integrator {

	ForwardEuler::ForwardEuler(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux) 
	: numericalFlux(numericalFlux) 
	{
		// Empty
	}


	///
	/// Returns the number of substeps this integrator uses.
	/// For ForwardEuler this is 1, for RK4 this is 4, etc.
	///
	size_t ForwardEuler::getNumberOfSubsteps() const {
		return 1;
	}

	///
	/// Performs one substep and stores the result to output.
	/// \note the next invocation to performSubstep will get as input the previuosly calculated outputs
	///
	real ForwardEuler::performSubstep(const std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
		rvec3 spatialCellSizes, real dt, real cfl,
        volume::Volume& output, size_t substep,
        const simulator::TimestepInformation& timestepInformation) {

		rvec3 waveSpeed(0, 0, 0);

        numericalFlux->computeFlux(*inputConserved[0], waveSpeed, true, output);
        dt = computeTimestep(waveSpeed, spatialCellSizes, cfl, timestepInformation);
		rvec3 cellScaling(dt / spatialCellSizes.x,
			dt / spatialCellSizes.y,
			dt / spatialCellSizes.z);

		output *= cellScaling.x;
        output += *inputConserved[0];

		return dt;

	}
}
}
