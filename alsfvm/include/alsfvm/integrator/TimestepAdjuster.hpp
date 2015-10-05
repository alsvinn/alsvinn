#pragma once
#include "alsfvm/simulator/TimestepInformation.hpp"

namespace alsfvm { namespace integrator { 

///
/// \brief The TimestepAdjuster class is an abstract interface for adjusting timesteps.
///
/// The main use is to ensure that we hit the save times exactly, ie. that if
/// the end time is T, then it will truncate the timestep to min(T - currentTime, dt)
///
/// \note The adjuster can only make the timestep SMALLER or equal to the previous given timestep.
///
///
    class TimestepAdjuster {
    public:

        ///
        /// \brief adjustTimestep returns the new timestep that the simulator should use
        /// \param dt the current timestep being used
        /// \param timestepInformation timesteps information
        /// \return the new timestep
        ///
        virtual real adjustTimestep(real dt, const simulator::TimestepInformation& timestepInformation) const = 0;
    };
} // namespace alsfvm
} // namespace integrator
