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
#include "alsutils/config.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/NumericalFluxCPU.hpp"

#include "alsfvm/reconstruction/ReconstructionFactory.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include "alsfvm/numflux/numerical_flux_list.hpp"

#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#endif


#include "alsutils/error/Exception.hpp"
#include <iostream>
#include <boost/algorithm/string.hpp>

namespace alsfvm {
namespace numflux {

namespace {

///
/// \brief The FluxFunctor struct is used to loop through every flux using
///        boost fusion. Note: We could use C++14-lambdas for this, but we
///        do not want to mix in C++14 yet.
///
template<class Equation>
struct FluxFunctor {
    FluxFunctor(const std::string& fluxName,
        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
        const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration,
        const grid::Grid& grid,
        alsfvm::shared_ptr<NumericalFlux>& numericalFlux)
        : fluxName(fluxName),
          reconstruction(reconstruction),
          simulatorParameters(simulatorParameters),
          deviceConfiguration(deviceConfiguration),
          grid(grid),
          numericalFlux(numericalFlux) {

    }

    template<class NumericalFlux>
    void operator()(const NumericalFlux& flux) const {
        if (NumericalFlux::name == boost::to_lower_copy(fluxName)) {
            if (deviceConfiguration->getPlatform() == "cpu") {
                if (grid.getActiveDimension() == 3) {
                    numericalFlux.reset(new NumericalFluxCPU<NumericalFlux, Equation, 3>(grid,
                            reconstruction, simulatorParameters, deviceConfiguration));
                } else if (grid.getActiveDimension() == 2) {
                    numericalFlux.reset(new NumericalFluxCPU<NumericalFlux, Equation, 2>(grid,
                            reconstruction, simulatorParameters, deviceConfiguration));
                } else if (grid.getActiveDimension() == 1) {
                    numericalFlux.reset(new NumericalFluxCPU<NumericalFlux, Equation, 1>(grid,
                            reconstruction, simulatorParameters, deviceConfiguration));
                } else {
                    THROW("Unsupported dimension " << grid.getActiveDimension());
                }
            }

#ifdef ALSVINN_HAVE_CUDA
            else if (deviceConfiguration->getPlatform() == "cuda") {
                if (grid.getActiveDimension() == 3) {
                    numericalFlux.reset(new NumericalFluxCUDA<NumericalFlux, Equation, 3>(grid,
                            reconstruction, *simulatorParameters, deviceConfiguration));
                } else if (grid.getActiveDimension() == 2) {
                    numericalFlux.reset(new NumericalFluxCUDA<NumericalFlux, Equation, 2>(grid,
                            reconstruction, *simulatorParameters, deviceConfiguration));
                } else if (grid.getActiveDimension() == 1) {
                    numericalFlux.reset(new NumericalFluxCUDA<NumericalFlux, Equation, 1>(grid,
                            reconstruction, *simulatorParameters, deviceConfiguration));
                } else {
                    THROW("Unsupported dimension " << grid.getActiveDimension());
                }
            }

#endif
            else {
                THROW("Unknown platform " << deviceConfiguration->getPlatform());
            }
        }
    }
    const std::string& fluxName;
    alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction;
    const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters;
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration;
    const grid::Grid& grid;
    alsfvm::shared_ptr<NumericalFlux>& numericalFlux;
};

///
/// \brief The EquationFunctor struct is used to loop through every equation using
///        boost fusion. Note: We could use C++14-lambdas for this, but we
///        do not want to mix in C++14 yet.
///
struct EquationFunctor {
    EquationFunctor(const std::string& equationName,
        const std::string& fluxName,
        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
        const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
        alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration,
        const grid::Grid& grid,
        alsfvm::shared_ptr<NumericalFlux>& numericalFlux)
        : equationName(equationName),
          fluxName(fluxName),
          reconstruction(reconstruction),
          simulatorParameters(simulatorParameters),
          deviceConfiguration(deviceConfiguration),
          grid(grid),
          numericalFlux(numericalFlux) {

    }

    template<class EquationInfo>
    void operator()(const EquationInfo& info) const {
        if (info.getName() == equationName) {
            FluxFunctor<typename EquationInfo::EquationType> fluxFunctor(fluxName,
                reconstruction, simulatorParameters,
                deviceConfiguration, grid, numericalFlux);
            for_each_flux<typename EquationInfo::EquationType> (fluxFunctor);
        }
    }

    const std::string& equationName;
    const std::string& fluxName;
    alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction;
    const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters;
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration;
    const grid::Grid& grid;
    alsfvm::shared_ptr<NumericalFlux>& numericalFlux;
};
}


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
      deviceConfiguration(deviceConfiguration),
      simulatorParameters(simulatorParameters) {
    // empty
}

///
/// Creates the numerical flux
///
NumericalFluxFactory::NumericalFluxPtr NumericalFluxFactory::createNumericalFlux(
    const grid::Grid& grid) {

    auto memoryFactory = alsfvm::make_shared<memory::MemoryFactory>
        (deviceConfiguration);

    alsfvm::reconstruction::ReconstructionFactory reconstructionFactory;
    auto reconstructor = reconstructionFactory.createReconstruction(reconstruction,
            equation,
            *simulatorParameters,
            memoryFactory,
            grid,
            deviceConfiguration);

    alsfvm::shared_ptr<NumericalFlux> numericalFlux;
    EquationFunctor equationFunctor(equation, fluxname, reconstructor,
        simulatorParameters,
        deviceConfiguration, grid, numericalFlux);

    alsfvm::equation::for_each_equation(equationFunctor);

    if (!numericalFlux) {
        THROW("Something went wrong in NumericalFluxFactory::createNumericalFlux. "
            "Check equation and flux.");
    }

    return numericalFlux;
}

}
}

