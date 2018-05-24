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
#include "alsfvm/numflux/NumericalFlux.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsfvm {
namespace numflux {

///
/// Use this to instantiate new numerical fluxes.
///
class NumericalFluxFactory {
public:
    ///
    /// The numerical flux pointer
    ///
    typedef alsfvm::shared_ptr < NumericalFlux > NumericalFluxPtr;

    ///
    /// \param equation the name of the equation (eg. Euler)
    /// \param fluxname the name of the flux (eg. HLL)
    /// \param reconstruction the reconstruction to use ("none" is default).
    /// \param deviceConfiguration the relevant device configuration
    /// \param simulatorParameters the given parameters for the simulation
    ///
    /// \note The platform name is deduced by deviceConfiguration
    ///
    NumericalFluxFactory(const std::string& equation,
        const std::string& fluxname,
        const std::string& reconstruction,
        const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

    ///
    /// Creates the numerical flux
    ///
    NumericalFluxPtr createNumericalFlux(const grid::Grid& grid);

private:
    std::string equation;
    std::string fluxname;
    std::string reconstruction;
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration;
    const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters;
};
} // namespace alsfvm
} // namespace numflux
