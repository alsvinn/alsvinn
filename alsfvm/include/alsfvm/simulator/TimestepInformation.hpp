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
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace simulator {

class TimestepInformation {
public:
    TimestepInformation(real currentTime, size_t numberOfStepsPerformed);
    TimestepInformation();

    ///
    /// \brief incrementTime increments the current simulation time
    /// \param dt the increment size.
    ///
    void incrementTime(real dt);

    ///
    /// \return the current simulation time
    ///
    real getCurrentTime() const;

    ///
    /// \brief getNumberOfStepsPerformed returns the number of timesteps calculated.
    ///
    size_t getNumberOfStepsPerformed() const;

private:
    real currentTime;
    size_t numberOfStepsPerformed;
};

} // namespace simulator
} // namespace alsfvm


