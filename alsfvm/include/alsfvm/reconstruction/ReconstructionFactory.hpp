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
#include "alsfvm/reconstruction/Reconstruction.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"
#include "alsfvm/grid/Grid.hpp"

namespace alsfvm {
namespace reconstruction {

//! Responsible for creating the different reconstructions
class ReconstructionFactory {
public:
    typedef alsfvm::shared_ptr<Reconstruction> ReconstructionPtr;

    //! Create the reconstruction.
    //!
    //! @param name the name of the reconstruction. Possibilities:
    //!             name   | description
    //!             -------+-------------------------------------------
    //!             none   | no reconstruction
    //!             eno2   | second order ENO
    //!             eno3   | third order ENO
    //!             eno4   | fourth order ENO
    //!             weno2  | second order WENO
    //!             weno3  | third order WENO
    //!             wenof2 | second order WENOF (clamping of variables)
    //!
    //! @param equation equation name. Currently only supports "euler1", "euler2", "euler3" and
    //!                 "burgers"
    //!
    //! @param simulatorParameters the parameters to be used (only used for WENOF)
    //!
    //! @param memoryFactory used to create new temporary memory areas (relevant for ENO)
    //!
    //! @param grid the grid to compute on
    //!
    //! @param deviceConfiguration the deviceConfiguration to use.
    ReconstructionPtr createReconstruction(const std::string& name,
        const std::string& equation,
        const simulator::SimulatorParameters& simulatorParameters,
        alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
        const grid::Grid& grid,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

};
}
}
