#include "alsfvm/integrator/ForwardEuler.hpp"

namespace alsfvm { namespace integrator {

	ForwardEuler::ForwardEuler(std::shared_ptr<numflux::NumericalFlux> numericalFlux) 
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
	void ForwardEuler::performSubstep(const volume::Volume& inputConserved, const volume::Volume& inputExtra,
		rvec3 spatialCellSizes, real dt,
		volume::Volume& output) {

		numericalFlux->computeFlux(inputConserved, inputExtra, spatialCellSizes, output);
		output *= dt;
		output += inputConserved;


	}
}
}
