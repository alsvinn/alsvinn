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

#include "alsfvm/diffusion/DiffusionFactory.hpp"
#include "alsfvm/reconstruction/tecno/ReconstructionFactory.hpp"
#include "alsfvm/diffusion/TecnoDiffusionCPU.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/diffusion/TecnoDiffusionCUDA.hpp"
#endif

#include "alsfvm/diffusion/RoeMatrix.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/diffusion/NoDiffusion.hpp"
#include "alsfvm/diffusion/RusanovMatrix.hpp"


namespace alsfvm {
namespace diffusion {
alsfvm::shared_ptr<DiffusionOperator> DiffusionFactory::createDiffusionOperator(
    const std::string& equation,
    const std::string& diffusionType,
    const std::string& reconstructionType,
    const grid::Grid& grid,
    const simulator::SimulatorParameters& simulatorParameters,
    alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration,
    alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
    volume::VolumeFactory& volumeFactory
) {
    reconstruction::tecno::ReconstructionFactory reconstructionFactory;
    auto reconstruction = reconstructionFactory.createReconstruction(
            reconstructionType, equation,
            simulatorParameters, memoryFactory, grid, deviceConfiguration);

    alsfvm::shared_ptr<DiffusionOperator> diffusionOperator;

    if (diffusionType == "none") {
        diffusionOperator.reset(new NoDiffusion());
    } else if (deviceConfiguration->getPlatform() == "cpu") {
        if (equation == "burgers") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::burgers::Burgers, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::burgers::Burgers, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else if (equation == "euler1") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::euler::Euler<1>, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::euler::Euler<1>, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else if (equation == "euler2") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::euler::Euler<2>, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::euler::Euler<2>, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else if (equation == "euler3") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::euler::Euler<3>, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCPU
                    <equation::euler::Euler<3>, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else {
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
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else if (equation == "euler1") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCUDA
                    <equation::euler::Euler<1>, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCUDA
                    <equation::euler::Euler<1>, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else if (equation == "euler2") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCUDA
                    <equation::euler::Euler<2>, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCUDA
                    <equation::euler::Euler<2>, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else if (equation == "euler3") {
            if (diffusionType == "tecnoroe") {
                diffusionOperator.reset(new TecnoDiffusionCUDA
                    <equation::euler::Euler<3>, RoeMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else if (diffusionType == "tecnorusanov") {
                diffusionOperator.reset(new TecnoDiffusionCUDA
                    <equation::euler::Euler<3>, RusanovMatrix>(volumeFactory, reconstruction,
                        simulatorParameters));
            } else {
                THROW("Unknown diffusion type " << diffusionType);
            }

        } else {
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
