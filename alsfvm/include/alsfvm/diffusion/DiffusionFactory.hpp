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
#include "alsfvm/diffusion/DiffusionOperator.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/grid/Grid.hpp"


namespace alsfvm {
namespace diffusion {

class DiffusionFactory {
public:
    alsfvm::shared_ptr<DiffusionOperator> createDiffusionOperator(
        const std::string& equation,
        const std::string& diffusionType,
        const std::string& reconstructionType,
        const grid::Grid& grid,
        const simulator::SimulatorParameters& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration,
        alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
        volume::VolumeFactory& volumeFactory
    );
};
} // namespace diffusion
} // namespace alsfvm
