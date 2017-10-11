#include "alsfvm/integrator/RungeKutta4.hpp"
#include "alsfvm/types.hpp"

namespace alsfvm {
    namespace integrator {


        RungeKutta4::RungeKutta4(alsfvm::shared_ptr<System> system)
            : system(system)
        {

        }


        ///
        /// Returns the number of substeps this integrator uses.
        /// Since this is third order RK, we need three subtimesteps
        ///
        /// \returns 3
        ///
        size_t RungeKutta4::getNumberOfSubsteps() const {
            return 4;
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
        real RungeKutta4::performSubstep(std::vector<alsfvm::shared_ptr<volume::Volume> > &inputConserved,
            rvec3 spatialCellSizes, real dt, real cfl,
            volume::Volume& output, size_t substep,
            const simulator::TimestepInformation& timestepInformation) {
            // We compute U + dt * F(U)


            // Compute F(U)
            rvec3 waveSpeeds(0, 0, 0);
            (*system)(*inputConserved[substep], waveSpeeds, true, output);

            if (substep == 0) {
                dt = computeTimestep(waveSpeeds, spatialCellSizes, cfl, timestepInformation);
            }

            
            
            rvec3 cellScaling(dt / spatialCellSizes.x,
                dt / spatialCellSizes.y,
                dt / spatialCellSizes.z);

            if (substep < 3) {
                output *= cellScaling.x;
            }
            if (substep < 2) {
                output *= 0.5;
                
            }
            
            if (substep < 3) {
                output += *inputConserved[0];
            }
            
            
            if (substep == 3) {
                output.addLinearCombination((1.0 /6.0)*cellScaling.x,
                    1.0 / 3.0, *inputConserved[1],
                    2.0 / 3.0, *inputConserved[2],
                    1.0 / 3.0, *inputConserved[3],
                    -1.0/3.0 , *inputConserved[0]);

            }

            return dt;
        }
    }
}


