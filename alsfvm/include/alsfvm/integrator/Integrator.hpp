#pragma once
#include "alsfvm/volume/Volume.hpp"
#include "alsfvm/integrator/TimestepAdjuster.hpp"
#include "alsfvm/integrator/WaveSpeedAdjuster.hpp"



namespace alsfvm {
namespace integrator {

///
/// Base interface for all integrators.
/// An integrator will solve the system
/// \f[u_t+Q(u)=0\f]
/// where the function \f$Q\f$ is supplied (usually a numerical flux).
///
/// We assume the time integrator is divided into a number of subtimesteps,
/// so that it can be run in the following manner
/// \code{.cpp}
/// // PSEUDOCODE!!!
/// // For each substep we need one buffer to hold the output
/// buffers = makeBuffers(integrator.getNumberOfSubsteps());
/// setupInput(buffers[0]);
/// const size_t numberOfSubsteps = integrator.getNumberOfSubsteps();
/// while(t < tEnd) {
///
///    for(size_t subStep = 0; subStep < numberOfSubsteps; subStep++) {
///
///        integrator.performSubstep(buffers[subStep].conserved(),
///                                  buffers[(subStep+1) % numberOfSubsteps].conserved(), substep);
///        buffers[(subStep+1) % numberOfSubsteps].computeExtra();
///     }
///     t += dt;
/// }
/// \endcode
///
class Integrator {
    public:
        virtual ~Integrator() {}
        ///
        /// Returns the number of substeps this integrator uses.
        /// For ForwardEuler this is 1, for RK4 this is 4, etc.
        ///
        virtual size_t getNumberOfSubsteps() const = 0;

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
        /// \param timestepInformation the current timestepInformation (needed for current time)
        /// \note the next invocation to performSubstep will get as input the previuosly calculated outputs
        /// \returns the newly computed timestep (each integrator may choose to change the timestep)
        ///
        virtual real performSubstep( std::vector<alsfvm::shared_ptr< volume::Volume> >&
            inputConserved,
            rvec3 spatialCellSizes, real dt, real cfl,
            volume::Volume& output, size_t substep,
            const simulator::TimestepInformation& timestepInformation) = 0;

        ///
        /// Computes the timestep (dt).
        /// \param[in] waveSpeeds the wave speeds in each direction
        /// \param[in] cellLengths the cell lengths in each direction
        /// \param[in] cfl the CFL number
        /// \param[in] timestepInformation the timestep information.
        ///
        real computeTimestep(const rvec3& waveSpeeds, const rvec3& cellLengths,
            real cfl,
            const simulator::TimestepInformation& timestepInformation) const;

        ///
        /// \brief addTimestepAdjuster adds a timestep adjuster
        /// \param adjuster the adjuster to add
        ///
        void addTimestepAdjuster(alsfvm::shared_ptr<TimestepAdjuster>& adjuster);

        void addWaveSpeedAdjuster(WaveSpeedAdjusterPtr adjuster);

    protected:
        ///
        /// \brief adjustTimestep adjusts the timesteps according to the timestepsadjusters
        /// \param dt the current timestep
        /// \param timestepInformation the current timestep information
        /// \return the new timestep (or dt if unchanged)
        ///
        real adjustTimestep(real dt,
            const simulator::TimestepInformation& timestepInformation) const;

    private:
        std::vector<alsfvm::shared_ptr<TimestepAdjuster> > timestepAdjusters;
        std::vector<WaveSpeedAdjusterPtr > waveSpeedAdjusters;

};

} // namespace alsfvm

} // namespace integrator

