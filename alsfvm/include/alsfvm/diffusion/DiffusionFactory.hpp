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
