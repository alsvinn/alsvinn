#pragma once
#include "alsfvm/integrator/Integrator.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"

namespace alsfvm { namespace integrator { 

    class ForwardEuler : public Integrator {
    public:
		///
		/// Constructs a new instance.
		///
		/// \param numericalFlux the numerical flux to use 
		///
		ForwardEuler(std::shared_ptr<numflux::NumericalFlux> numericalFlux);

		///
		/// Returns the number of substeps this integrator uses.
		/// For ForwardEuler this is 1, for RK4 this is 4, etc.
		///
		virtual size_t getNumberOfSubsteps() const;

		///
		/// Performs one substep and stores the result to output.
		/// \note the next invocation to performSubstep will get as input the previuosly calculated outputs
		///
		virtual void performSubstep(const volume::Volume& inputConserved, const volume::Volume& inputExtra,
			rvec3 spatialCellSizes, real dt,
			volume::Volume& output);

	private:
		std::shared_ptr<numflux::NumericalFlux> numericalFlux;
    };
} // namespace alsfvm
} // namespace integrator

