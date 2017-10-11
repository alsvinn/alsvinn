#include "alsfvm/integrator/ForwardEuler.hpp"
#include <iostream>
namespace alsfvm { namespace integrator {

    ForwardEuler::ForwardEuler(alsfvm::shared_ptr<System> system)
    : system(system)
	{
		// Empty
	}


	size_t ForwardEuler::getNumberOfSubsteps() const {
		return 1;
	}


    real ForwardEuler::performSubstep( std::vector<alsfvm::shared_ptr< volume::Volume> >& inputConserved,
		rvec3 spatialCellSizes, real dt, real cfl,
        volume::Volume& output, size_t substep,
        const simulator::TimestepInformation& timestepInformation) {

		rvec3 waveSpeed(0, 0, 0);

        (*system)(*inputConserved[0], waveSpeed, true, output);
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
