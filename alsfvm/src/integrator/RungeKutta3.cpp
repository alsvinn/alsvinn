#include "alsfvm/integrator/RungeKutta3.hpp"

namespace alsfvm { namespace integrator {


RungeKutta3::RungeKutta3(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux)
    : numericalFlux(numericalFlux)
{

}


///
/// Returns the number of substeps this integrator uses.
/// Since this is third order RK, we need three subtimesteps
///
/// \returns 3
///
size_t RungeKutta3::getNumberOfSubsteps() const {
    return 3;
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
void RungeKutta3::performSubstep(const std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
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
        output *= 1./3.;
        output += *inputConserved[0];
        output *= 3./4.;
    } else if (substep == 2) {
        output *= 2;
        output += *inputConserved[0];
        output *= 1./3.;
    }
}
}
}


