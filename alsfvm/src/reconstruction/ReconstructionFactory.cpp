#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#include "alsfvm/reconstruction/WENOCUDA.hpp"
#include "alsfvm/reconstruction/WENO2CUDA.hpp"
#include "alsfvm/reconstruction/NoReconstructionCUDA.hpp"
#endif
#include "alsfvm/reconstruction/WENOF2.hpp"
#include "alsfvm/reconstruction/WENO2.hpp"
#include "alsfvm/reconstruction/ReconstructionCPU.hpp"
#include "alsfvm/reconstruction/ReconstructionCUDA.hpp"
#include "alsfvm/reconstruction/NoReconstruction.hpp"
#include "alsfvm/reconstruction/ENOCPU.hpp"

#include "alsfvm/reconstruction/WENOCPU.hpp"

#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace reconstruction {

ReconstructionFactory::ReconstructionPtr
    ReconstructionFactory::createReconstruction(const std::string &name,
                                                const std::string &equation,
                                                const simulator::SimulatorParameters& simulatorParameters,
                                                alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                                                const grid::Grid& grid,
                                                alsfvm::shared_ptr<DeviceConfiguration> &deviceConfiguration)
{
    if (equation != "euler") {
        THROW("Unknown equation " << equation);
    }
    auto& platform = deviceConfiguration->getPlatform();
    alsfvm::shared_ptr<reconstruction::Reconstruction> reconstructor;
    if (platform == "cpu") {
        if (name == "none") {
            reconstructor.reset(new reconstruction::NoReconstruction);
        }
        else if (name == "eno2") {
            reconstructor.reset(new reconstruction::ENOCPU<2>(memoryFactory, grid.getDimensions().x,
                grid.getDimensions().y,
                grid.getDimensions().z));

        }
        else if (name == "eno3") {
            reconstructor.reset(new reconstruction::ENOCPU<3>(memoryFactory, grid.getDimensions().x,
                grid.getDimensions().y,
                grid.getDimensions().z));

        }
        else if (name == "eno4") {
            reconstructor.reset(new reconstruction::ENOCPU<4>(memoryFactory, grid.getDimensions().x,
                grid.getDimensions().y,
                grid.getDimensions().z));

        }
        else if (name == "weno2") {
            reconstructor.reset(new reconstruction::ReconstructionCPU<reconstruction::WENO2<equation::euler::Euler>, equation::euler::Euler>(simulatorParameters));

        }
        else if (name == "weno3") {
            reconstructor.reset(new reconstruction::WENOCPU<3>());

        }
        else if (name == "wenof2") {
            reconstructor.reset(new reconstruction::ReconstructionCPU<reconstruction::WENOF2<equation::euler::Euler>, equation::euler::Euler>(simulatorParameters));

        }

        else {
            THROW("Unknown reconstruction " << name);
        }
    }
#ifdef ALSVINN_HAVE_CUDA
    else if (platform == "cuda") {
        if (name == "none") {
            reconstructor.reset(new reconstruction::NoReconstructionCUDA);
        }
        else if (name == "eno2") {
            THROW("eno2 not supported on CUDA at the moment.")

        }
        else if (name == "eno3") {
            THROW("eno3 not supported on CUDA at the moment.")

        }
        else if (name == "eno4") {
            THROW("eno4 not supported on CUDA at the moment.")
        }
        else if (name == "weno2") {
            if (equation == "euler") {
                //reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
                reconstructor.reset(new reconstruction::ReconstructionCUDA<reconstruction::WENO2<equation::euler::Euler>, equation::euler::Euler>(simulatorParameters));

            }
            else {
                THROW("We do not support WENOCUDA for equation " << equation);
            }

        }
        else if (name == "weno3") {
            if (equation == "euler") {
                reconstructor.reset(new reconstruction::WENOCUDA<equation::euler::Euler, 3>());
            }
            else {
                THROW("We do not support WENOCUDA for equation " << equation);
            }

        }
        else if (name == "wenof2") {
            reconstructor.reset(new reconstruction::ReconstructionCUDA<reconstruction::WENOF2<equation::euler::Euler>, equation::euler::Euler>(simulatorParameters));

        }
        else {
            THROW("Unknown reconstruction " << reconstruction);
        }

    }
#endif
    else {
        THROW("Unknown platform " << platform);
    }

    return reconstructor;
}

}
}
