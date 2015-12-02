#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"

#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"

#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#endif


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

    alsfvm::reconstruction::ReconstructionFactory reconstructionFactory;
    auto reconstructor = reconstructionFactory.createReconstruction(reconstruction,
                                                                    equation,
                                                                    *simulatorParameters,
                                                                    memoryFactory,
                                                                    grid,
                                                                    deviceConfiguration);

    // First we must do a lot of error checking
    auto& platform = deviceConfiguration->getPlatform();
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

