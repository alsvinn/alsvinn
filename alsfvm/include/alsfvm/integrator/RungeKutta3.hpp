#pragma once
#include "alsfvm/integrator/Integrator.hpp"
#include "alsfvm/numflux/NumericalFlux.hpp"

namespace alsfvm { namespace integrator {

    ///
    /// Does 3rd order RungeKutta-integrator. In other words,
    /// \todo Update documentation
    ///
    class RungeKutta3 : public Integrator {
    public:
		RungeKutta3(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux);


        ///
        /// Returns the number of substeps this integrator uses.
        /// For ForwardEuler this is 1, for RK4 this is 4, etc.
        ///
        /// \returns 3
        ///
        virtual size_t getNumberOfSubsteps() const;

		///
		/// Performs one substep and stores the result to output.
		///
		/// \param inputConserved should have the output from the previous invocations
		///        in this substep, if this is the first invocation, then this will have one element,
		///        second timestep 2 elements, etc.
		/// \param spatialCellSizes should be the cell size in each direction
		/// \param dt is the timestep
		/// \param substep is the currently computed substep, starting at 0.
		/// \param output where to write the output
		/// \param cfl the cfl number to use.
		/// \note the next invocation to performSubstep will get as input the previuosly calculated outputs
		/// \returns the newly computed timestep (each integrator may choose to change the timestep)
		///
		virtual real performSubstep(const std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
			rvec3 spatialCellSizes, real dt, real cfl,
			volume::Volume& output, size_t substep);

    private:
		alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux;

    };
} // namespace alsfvm
} // namespace integrator
