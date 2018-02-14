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
