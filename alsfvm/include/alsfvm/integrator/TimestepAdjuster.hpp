/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "alsfvm/simulator/TimestepInformation.hpp"

namespace alsfvm {
namespace integrator {

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
    virtual real adjustTimestep(real dt,
        const simulator::TimestepInformation& timestepInformation) const = 0;
};
} // namespace alsfvm
} // namespace integrator
