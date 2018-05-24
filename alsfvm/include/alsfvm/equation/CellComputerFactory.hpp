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
#include <memory>
#include "alsfvm/equation/CellComputer.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"

namespace alsfvm {
namespace equation {

///
/// \brief The CellComputerFactory class is used to create new cell computers
///
class CellComputerFactory {
public:
    ///
    /// \brief CellComputerFactory construct a new factory instance
    /// \param parameters the relevant simulatorParameters.
    /// \param deviceConfiguration the deviceConfiguration used.
    ///
    CellComputerFactory(const alsfvm::shared_ptr<simulator::SimulatorParameters>&
        parameters,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

    ///
    /// \brief createComputer creates a new cell computer
    /// \return an instance of the cell computer.
    ///
    alsfvm::shared_ptr<CellComputer> createComputer();

private:
    const alsfvm::shared_ptr<simulator::SimulatorParameters> simulatorParameters;

};
} // namespace alsfvm
} // namespace equation
