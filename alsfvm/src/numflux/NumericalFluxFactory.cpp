#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#include "alsfvm/reconstruction/WENOCUDA.hpp"
#include "alsfvm/reconstruction/WENO2CUDA.hpp"
#include "alsfvm/reconstruction/NoReconstructionCUDA.hpp"
#endif
#include "alsfvm/reconstruction/WENOF2.hpp"
#include "alsfvm/reconstruction/ReconstructionCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include "alsfvm/reconstruction/NoReconstruction.hpp"
#include "alsfvm/reconstruction/ENOCPU.hpp"

#include "alsfvm/reconstruction/WENOCPU.hpp"
#include "alsfvm/error/Exception.hpp"
#include <iostream>

namespace alsfvm { namespace numflux { 



///
/// \param equation the name of the equation (eg. Euler)
/// \param fluxname the name of the flux (eg. HLL)
/// \param reconstruction the reconstruction to use ("none" is default).
/// \param deviceConfiguration the relevant device configuration
/// \note The platform name is deduced by deviceConfiguration
///
NumericalFluxFactory::NumericalFluxFactory(const std::string& equation,
                                           const std::string& fluxname,
                                           const std::string& reconstruction,
                                           const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
                                           alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : equation(equation), fluxname(fluxname), reconstruction(reconstruction),
      deviceConfiguration(deviceConfiguration), simulatorParameters(simulatorParameters)
{
    // empty
}

///
/// Creates the numerical flux
///
NumericalFluxFactory::NumericalFluxPtr
NumericalFluxFactory::createNumericalFlux(const grid::Grid& grid) {
    auto memoryFactory = alsfvm::make_shared<memory::MemoryFactory>(deviceConfiguration);
    // First we must do a lot of error checking
    auto& platform = deviceConfiguration->getPlatform();
	alsfvm::shared_ptr<reconstruction::Reconstruction> reconstructor;
	if (platform == "cpu") {
		if (reconstruction == "none") {
			reconstructor.reset(new reconstruction::NoReconstruction);
		}
		else if (reconstruction == "eno2") {
			reconstructor.reset(new reconstruction::ENOCPU<2>(memoryFactory, grid.getDimensions().x,
				grid.getDimensions().y,
				grid.getDimensions().z));

		}
		else if (reconstruction == "eno3") {
			reconstructor.reset(new reconstruction::ENOCPU<3>(memoryFactory, grid.getDimensions().x,
				grid.getDimensions().y,
				grid.getDimensions().z));

		}
		else if (reconstruction == "eno4") {
			reconstructor.reset(new reconstruction::ENOCPU<4>(memoryFactory, grid.getDimensions().x,
				grid.getDimensions().y,
				grid.getDimensions().z));

		}
		else if (reconstruction == "weno2") {
			reconstructor.reset(new reconstruction::WENOCPU<2>());

		}
		else if (reconstruction == "weno3") {
			reconstructor.reset(new reconstruction::WENOCPU<3>());

		}
        else if (reconstruction == "wenof2") {
            reconstructor.reset(new reconstruction::ReconstructionCPU<reconstruction::WENOF2<equation::euler::Euler>, equation::euler::Euler>(*simulatorParameters));

        }

		else {
			THROW("Unknown reconstruction " << reconstruction);
		}
	}
#ifdef ALSVINN_HAVE_CUDA
	else if (platform == "cuda") {
		if (reconstruction == "none") {
			reconstructor.reset(new reconstruction::NoReconstructionCUDA);
		}
		else if (reconstruction == "eno2") {
			THROW("eno2 not supported on CUDA at the moment.")

		}
		else if (reconstruction == "eno3") {
			THROW("eno3 not supported on CUDA at the moment.")

		}
		else if (reconstruction == "eno4") {
			THROW("eno4 not supported on CUDA at the moment.")
		}
		else if (reconstruction == "weno2") {
			if (equation == "euler") {
				reconstructor.reset(new reconstruction::WENO2CUDA<equation::euler::Euler>());
			}
			else {
				THROW("We do not support WENOCUDA for equation " << equation);
			}

		}
		else if (reconstruction == "weno3") {
			if (equation == "euler") {
				reconstructor.reset(new reconstruction::WENOCUDA<equation::euler::Euler, 3>());
			}
			else {
				THROW("We do not support WENOCUDA for equation " << equation);
			}

		}
		else {
			THROW("Unknown reconstruction " << reconstruction);
		}
		
	}
#endif
    else {
        THROW("Unknown platform " << platform);
    }
	if (platform == "cpu") {
		if (equation == "euler") {
			if (fluxname == "HLL") {

				if (grid.getActiveDimension() == 3) {
                    return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL, 3>(grid, reconstructor, simulatorParameters, deviceConfiguration));
				}
				else if (grid.getActiveDimension() == 2) {
                    return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL, 2>(grid, reconstructor, simulatorParameters, deviceConfiguration));
				}
				else if (grid.getActiveDimension() == 1) {
                    return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL, 1>(grid, reconstructor, simulatorParameters, deviceConfiguration));
				}
				else {
					THROW("Unsupported dimension " << grid.getActiveDimension()
						<< " for equation " << equation << " platform " << platform << " and fluxname " << fluxname);
				}

			}
			else if (fluxname == "HLL3") {
				if (fluxname == "HLL3") {

					if (grid.getActiveDimension() == 3) {
                        return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL3, 3>(grid, reconstructor, simulatorParameters, deviceConfiguration));
					}
					else if (grid.getActiveDimension() == 2) {
                        return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL3, 2>(grid, reconstructor, simulatorParameters, deviceConfiguration));
					}
					else if (grid.getActiveDimension() == 1) {
                        return NumericalFluxPtr(new euler::NumericalFluxCPU<euler::HLL3, 1>(grid, reconstructor, simulatorParameters, deviceConfiguration));
					}
					else {
						THROW("Unsupported dimension " << grid.getActiveDimension()
							<< " for equation " << equation << " platform " << platform << " and fluxname " << fluxname);
					}

				}
			}
			else {
				THROW("Unknown flux " << fluxname);
			}
		}
		else {
			THROW("Unknown equation " << equation);
		}
	}
#ifdef ALSVINN_HAVE_CUDA
	else if (platform == "cuda") {
		if (equation == "euler") {
			if (fluxname == "HLL") {

				if (grid.getActiveDimension() == 3) {
                    return NumericalFluxPtr(new NumericalFluxCUDA<euler::HLL, equation::euler::Euler, 3>(grid, reconstructor, *simulatorParameters, deviceConfiguration));
				}
				else if (grid.getActiveDimension() == 2) {
                    return NumericalFluxPtr(new NumericalFluxCUDA<euler::HLL, equation::euler::Euler, 2>(grid, reconstructor, *simulatorParameters, deviceConfiguration));
				}
				else if (grid.getActiveDimension() == 1) {
                    return NumericalFluxPtr(new NumericalFluxCUDA<euler::HLL, equation::euler::Euler, 1>(grid, reconstructor, *simulatorParameters, deviceConfiguration));
				}
				else {
					THROW("Unsupported dimension " << grid.getActiveDimension()
						<< " for equation " << equation << " platform " << platform << " and fluxname " << fluxname);
				}

			}
			else if (fluxname == "HLL3") {
				if (fluxname == "HLL3") {

					if (grid.getActiveDimension() == 3) {
						return NumericalFluxPtr(new NumericalFluxCUDA<euler::HLL3, equation::euler::Euler, 3>(grid, reconstructor, *simulatorParameters, deviceConfiguration));
					}
					else if (grid.getActiveDimension() == 2) {
                        return NumericalFluxPtr(new NumericalFluxCUDA<euler::HLL3, equation::euler::Euler, 2>(grid, reconstructor, *simulatorParameters, deviceConfiguration));
					}
					else if (grid.getActiveDimension() == 1) {
                        return NumericalFluxPtr(new NumericalFluxCUDA<euler::HLL3, equation::euler::Euler, 1>(grid, reconstructor, *simulatorParameters, deviceConfiguration));
					}
					else {
						THROW("Unsupported dimension " << grid.getActiveDimension()
							<< " for equation " << equation << " platform " << platform << " and fluxname " << fluxname);
					}

				}
			}
			else {
				THROW("Unknown flux " << fluxname);
			}
		}
		else {
			THROW("Unknown equation " << equation);
		}
    }
#endif
    else {
        THROW("Unknown platform " << platform);
    }

    THROW("Something went wrong in NumericalFluxFactory::createNumericalFlux");
}

}}

