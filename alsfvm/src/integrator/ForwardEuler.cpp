#include "alsfvm/integrator/ForwardEuler.hpp"

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
    void ForwardEuler::performSubstep(const std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
		rvec3 spatialCellSizes, real dt,
        volume::Volume& output, size_t substep) {

        rvec3 cellScaling(dt/spatialCellSizes.x,
                          dt/spatialCellSizes.y,
                          dt/spatialCellSizes.z);

        numericalFlux->computeFlux(*inputConserved[0], cellScaling, output);
        output += *inputConserved[0];


	}
}
}
