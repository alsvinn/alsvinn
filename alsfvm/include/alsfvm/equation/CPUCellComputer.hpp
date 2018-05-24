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
#include "alsfvm/equation/CellComputer.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
namespace alsfvm {
namespace equation {

template<class Equation>
class CPUCellComputer : public CellComputer {
public:
    CPUCellComputer(simulator::SimulatorParameters& parameters);
    ///
    /// \brief computeExtraVariables computes the extra variables (eg. pressure for euler)
    /// \param[in] conservedVariables the conserved variables to read from
    /// \param[out] extraVariables the extra variables to write to
    ///
    virtual void computeExtraVariables(const volume::Volume& conservedVariables,
        volume::Volume& extraVariables);

    ///
    /// Computes the maximum wavespeed
    /// \param conservedVariables the conserved variables (density, momentum, Energy for Euler)
    /// \param extraVariables the extra variables (pressure and velocity for Euler)
    /// \param direction the direction to find the wave speed for
    ///        direction | description
    ///        ----------|------------
    ///            0     |  x-direction
    ///            1     |  y-direction
    ///            2     |  z-direction
    /// \return the maximum wave speed (absolute value)
    ///
    virtual real computeMaxWaveSpeed(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables, size_t direction);

    ///
    /// Checks if all the constraints for the equation are met
    /// \param conservedVariables the conserved variables (density, momentum, Energy for Euler)
    /// \param extraVariables the extra variables (pressure and velocity for Euler)
    /// \return true if it obeys the constraints, false otherwise
    ///
    virtual bool obeysConstraints(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables);

    ///
    /// \brief computeFromPrimitive computes the conserved and extra variables based
    ///                             on the primtive variables
    /// \param[in] primtiveVariables the primitive variables to use
    /// \param[out] conservedVariables the conserved variables.
    /// \param[out] extraVariables the extra variables.
    ///
    virtual void computeFromPrimitive(const volume::Volume& primtiveVariables,
        volume::Volume& conservedVariables,
        volume::Volume& extraVariables);

private:
    typename Equation::Parameters parameters;
};
} // namespace alsfvm
} // namespace equation
