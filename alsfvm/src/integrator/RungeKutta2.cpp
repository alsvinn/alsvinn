#include "alsfvm/integrator/RungeKutta2.hpp"

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

///
/// Performs one substep and stores the result to output.
///
/// \param inputConserved should have the output from the previous invocations
///        in this substep, if this is the first invocation, then this will have one element,
///        second timestep 2 elements, etc.
/// \param spatialCellSizes should be the cell size in each direction
/// \param dt is the timestep
/// \param output where to write the output
/// \note the next invocation to performSubstep will get as input the previuosly calculated outputs
///
void RungeKutta2::performSubstep(const std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
    rvec3 spatialCellSizes, real dt,
    volume::Volume& output, size_t substep) {
    // We compute U + dt * F(U)

    rvec3 cellScaling(dt/spatialCellSizes.x,
                      dt/spatialCellSizes.y,
                      dt/spatialCellSizes.z);
    // Compute F(U)

    numericalFlux->computeFlux(*inputConserved[substep], cellScaling, output);
    output += *inputConserved[substep];

    if (substep == 1) {
        // 0.5 * (U+U')
        output += *inputConserved[0];
        output *= 0.5;
    }
}
}
}
