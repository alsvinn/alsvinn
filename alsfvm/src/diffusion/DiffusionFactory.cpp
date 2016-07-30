#include "alsfvm/diffusion/DiffusionFactory.hpp"
#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/diffusion/TecnoDiffusionCPU.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"
#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm { namespace diffusion { 
    alsfvm::shared_ptr<DiffusionOperator> DiffusionFactory::createDiffusionOperator(const std::string& equation,
        const std::string& diffusionType,
        const std::string& reconstructionType,
        const grid::Grid& grid,
        const simulator::SimulatorParameters& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration,
        alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
        volume::VolumeFactory& volumeFactory
        )
    {
        reconstruction::ReconstructionFactory reconstructionFactory;
        auto reconstruction = reconstructionFactory.createReconstruction(reconstructionType, equation,
            simulatorParameters, memoryFactory, grid, deviceConfiguration);
        
        alsfvm::shared_ptr<DiffusionOperator> diffusionOperator;
        if (deviceConfiguration->getPlatform() == "cpu") {
            if (equation == "burgers") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new DiffusionOperator
                        <equation::burgers::Burgers, RoeMatrix<equation::burgers::Burgers>>(volumeFactory, reconstructionFactory,
                            simulatorParameters,
                            )
                }
            }
        }
    }
}
}
