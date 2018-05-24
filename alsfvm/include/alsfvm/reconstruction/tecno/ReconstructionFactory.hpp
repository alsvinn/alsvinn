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
#include "alsfvm/reconstruction/tecno/TecnoReconstruction.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/grid/Grid.hpp"
namespace alsfvm {
namespace reconstruction {
namespace tecno {

class ReconstructionFactory {
public:

    alsfvm::shared_ptr<TecnoReconstruction> createReconstruction(
        const std::string& name,
        const std::string& equation,
        const simulator::SimulatorParameters& simulatorParameters,
        alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
        const grid::Grid& grid,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration
    );

};
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
