#include "alsfvm/diffusion/DiffusionFactory.hpp"
#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/diffusion/TecnoDiffusionCPU.hpp"
#include "alsfvm/diffusion/TecnoDiffusionCUDA.hpp"

#include "alsfvm/diffusion/RoeMatrix.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/diffusion/NoDiffusion.hpp"
#include "alsfvm/diffusion/RusanovMatrix.hpp"

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
        auto reconstruction = reconstructionFactory.createReconstruction(reconstructionType, "burgers",
            simulatorParameters, memoryFactory, grid, deviceConfiguration);
        
        alsfvm::shared_ptr<DiffusionOperator> diffusionOperator;
        if (diffusionType == "none") {
            diffusionOperator.reset(new NoDiffusion());
        }
        else if (deviceConfiguration->getPlatform() == "cpu") {
            if (equation == "burgers") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::burgers::Burgers, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::burgers::Burgers, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else if (equation == "euler1") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::euler::Euler<1>, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::euler::Euler<1>, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else if (equation == "euler2") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::euler::Euler<2>, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::euler::Euler<2>, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else if (equation == "euler3") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::euler::Euler<3>, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCPU
                        <equation::euler::Euler<3>, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else {
                THROW("Equation not supported: " << equation);
            }
        }
#ifdef ALSVINN_HAVE_CUDA
        else if (deviceConfiguration->getPlatform() == "cuda") {
            if (equation == "burgers") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::burgers::Burgers, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::burgers::Burgers, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else if (equation == "euler1") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::euler::Euler<1>, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::euler::Euler<1>, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else if (equation == "euler2") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::euler::Euler<2>, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::euler::Euler<2>, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else if (equation == "euler3") {
                if (diffusionType == "tecnoroe") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::euler::Euler<3>, RoeMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                } else if (diffusionType == "tecnorusanov") {
                    diffusionOperator.reset(new TecnoDiffusionCUDA
                        <equation::euler::Euler<3>, RusanovMatrix>(volumeFactory, reconstruction,
                            simulatorParameters));
                }
                else {
                    THROW("Unknown diffusion type " << diffusionType);
                }

            }
            else {
                THROW("Equation not supported: " << equation);
            }
        }
#endif
        else {
            THROW("Platform not supported: " << deviceConfiguration->getPlatform());
        }

        return diffusionOperator;
    }
}
}
