#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/equation/equation_list.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/reconstruction/NoReconstructionCUDA.hpp"
    #include "alsfvm/reconstruction/ReconstructionCUDA.hpp"
    #include "alsfvm/reconstruction/ENOCUDA.hpp"

#endif
#include "alsfvm/reconstruction/WENOF2.hpp"
#include "alsfvm/reconstruction/MC.hpp"
#include "alsfvm/reconstruction/WENO2.hpp"
#include "alsfvm/reconstruction/ReconstructionCPU.hpp"
#include "alsfvm/reconstruction/ReconstructionCUDA.hpp"
#include "alsfvm/reconstruction/NoReconstruction.hpp"
#include "alsfvm/reconstruction/ENOCPU.hpp"

#include "alsfvm/reconstruction/WENOCPU.hpp"

#include "alsutils/error/Exception.hpp"

namespace alsfvm {
namespace reconstruction {

ReconstructionFactory::ReconstructionPtr
ReconstructionFactory::createReconstruction(
    const std::string& name,
    const std::string& equation,
    const simulator::SimulatorParameters& simulatorParameters,
    alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
    const grid::Grid& grid,
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration) {
    if (equation != "euler1" && equation != "euler2" && equation != "euler3"
        && equation != "burgers" && equation != "buckleyleverett"
        && equation != "cubic") {
        THROW("Unknown equation " << equation);
    }

    auto& platform = deviceConfiguration->getPlatform();
    alsfvm::shared_ptr<reconstruction::Reconstruction> reconstructor;

    if (platform == "cpu") {
        if (name == "none") {
            reconstructor.reset(new reconstruction::NoReconstruction);
        } else if (name == "eno2") {
            reconstructor.reset(new reconstruction::ENOCPU<2>(memoryFactory,
                    grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "eno3") {
            reconstructor.reset(new reconstruction::ENOCPU<3>(memoryFactory,
                    grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "eno4") {
            reconstructor.reset(new reconstruction::ENOCPU<4>(memoryFactory,
                    grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "weno2") {
            if (equation == "euler1") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENO2<equation::euler::Euler<1> >, equation::euler::Euler<1> >
                    (simulatorParameters));
            } else  if (equation == "euler2") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENO2<equation::euler::Euler<2> >, equation::euler::Euler<2> >
                    (simulatorParameters));
            } else  if (equation == "euler3") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENO2<equation::euler::Euler<3> >, equation::euler::Euler<3> >
                    (simulatorParameters));
            } else {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENO2<equation::burgers::Burgers>, equation::burgers::Burgers>
                    (simulatorParameters));
            }

        } else if (name == "weno3") {
            reconstructor.reset(new reconstruction::WENOCPU<3>());

        } else if (name == "wenof2") {
            if (equation == "euler1") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENOF2<equation::euler::Euler<1> >, equation::euler::Euler<1> >
                    (simulatorParameters));
            } else if (equation == "euler2") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENOF2<equation::euler::Euler<2> >, equation::euler::Euler<2> >
                    (simulatorParameters));
            } else if (equation == "euler3") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::WENOF2<equation::euler::Euler<3> >, equation::euler::Euler<3> >
                    (simulatorParameters));
            } else {
                THROW("We do not support WENO2F for Burgers.");
            }

        } else if (name == "mc") {

            if (equation == "euler1") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::MC<equation::euler::Euler<1> >, equation::euler::Euler<1> >
                    (simulatorParameters));
            } else  if (equation == "euler2") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::MC<equation::euler::Euler<2> >, equation::euler::Euler<2> >
                    (simulatorParameters));
            } else  if (equation == "euler3") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::MC<equation::euler::Euler<3> >, equation::euler::Euler<3> >
                    (simulatorParameters));
            } else {
                reconstructor.reset(new
                    reconstruction::ReconstructionCPU<reconstruction::MC<equation::burgers::Burgers>, equation::burgers::Burgers>
                    (simulatorParameters));
            }


        }

        else {
            THROW("Unknown reconstruction " << name);
        }
    }

#ifdef ALSVINN_HAVE_CUDA
    else if (platform == "cuda") {
        if (name == "none") {
            reconstructor.reset(new reconstruction::NoReconstructionCUDA);
        } else if (name == "eno2") {
            if (equation == "euler1") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<1>, 2>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "euler2") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<2>, 2>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "euler3") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<3>, 2>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "burgers") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::burgers::Burgers, 2>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else {
                THROW("We do not support ENO on CUDA for equation: " << equation);
            }
        } else if (name == "eno3") {
            if (equation == "euler1") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<1>, 3>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "euler2") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<2>, 3>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "euler3") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<3>, 3>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "burgers") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::burgers::Burgers, 3>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else {
                THROW("We do not support ENO on CUDA for equation: " << equation);
            }
        } else if (name == "eno4") {
            if (equation == "euler1") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<1>, 4>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "euler2") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<2>, 4>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "euler3") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::euler::Euler<3>, 4>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else if (equation == "burgers") {
                reconstructor.reset(new reconstruction::ENOCUDA<equation::burgers::Burgers, 4>
                    (memoryFactory, grid.getDimensions().x,
                        grid.getDimensions().y,
                        grid.getDimensions().z));
            } else {
                THROW("We do not support ENO on CUDA for equation: " << equation);
            }
        } else if (name == "weno2") {
            if (equation == "euler1") {
                //reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENO2<equation::euler::Euler<1>>, equation::euler::Euler<1> >
                    (simulatorParameters));

            } else if (equation == "euler2") {
                //reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENO2<equation::euler::Euler<2>>, equation::euler::Euler<2> >
                    (simulatorParameters));

            } else if (equation == "euler3") {
                //reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENO2<equation::euler::Euler<3>>, equation::euler::Euler<3> >
                    (simulatorParameters));

            } else if (equation == "burgers") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENO2<equation::burgers::Burgers>, equation::burgers::Burgers>
                    (simulatorParameters));
            } else {
                THROW("We do not support WENOCUDA for equation " << equation);
            }

        } else if (name == "weno3") {
            THROW("We currently do not support WENO3. Use weno2.");

        } else if (name == "wenof2") {
            if (equation == "euler1") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENOF2<equation::euler::Euler<1> >, equation::euler::Euler<1> >
                    (simulatorParameters));
            } else if (equation == "euler2") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENOF2<equation::euler::Euler<2> >, equation::euler::Euler<2> >
                    (simulatorParameters));
            } else if (equation == "euler3") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::WENOF2<equation::euler::Euler<3> >, equation::euler::Euler<3> >
                    (simulatorParameters));
            } else {
                THROW("We do not support WENO2F for Burgers.");
            }

        } else if (name == "mc") {
            if (equation == "euler1") {

                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::MC<equation::euler::Euler<1>>, equation::euler::Euler<1> >
                    (simulatorParameters));

            } else if (equation == "euler2") {
                //reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::MC<equation::euler::Euler<2>>, equation::euler::Euler<2> >
                    (simulatorParameters));

            } else if (equation == "euler3") {
                //reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::MC<equation::euler::Euler<3>>, equation::euler::Euler<3> >
                    (simulatorParameters));

            } else if (equation == "burgers") {
                reconstructor.reset(new
                    reconstruction::ReconstructionCUDA<reconstruction::MC<equation::burgers::Burgers>, equation::burgers::Burgers>
                    (simulatorParameters));
            } else {
                THROW("We do not support WENOCUDA for equation " << equation);
            }

        } else {
            THROW("Unknown reconstruction " << name);
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
