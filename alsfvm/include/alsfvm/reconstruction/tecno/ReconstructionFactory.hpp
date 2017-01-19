#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/reconstruction/tecno/TecnoReconstruction.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/grid/Grid.hpp"
namespace alsfvm { namespace reconstruction { namespace tecno { 

    class ReconstructionFactory {
    public:

        alsfvm::shared_ptr<TecnoReconstruction> createReconstruction(const std::string &name,
                                                                   const std::string &equation,
                                                                   const simulator::SimulatorParameters& simulatorParameters,
                                                                   alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                                                                   const grid::Grid& grid,
                                                                   alsfvm::shared_ptr<DeviceConfiguration> &deviceConfiguration
                                                                   );

    };
} // namespace tecno
} // namespace reconstruction
} // namespace alsfvm
