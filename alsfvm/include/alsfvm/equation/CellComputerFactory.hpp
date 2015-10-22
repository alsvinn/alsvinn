#pragma once
#include <memory>
#include "alsfvm/equation/CellComputer.hpp"
#include "alsfvm/DeviceConfiguration.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"

namespace alsfvm { namespace equation { 

    ///
    /// \brief The CellComputerFactory class is used to create new cell computers
    ///
    class CellComputerFactory {
    public:
        ///
        /// \brief CellComputerFactory construct a new factory instance
        /// \param platform the platform (eg. "cpu", "cuda")
        /// \param equation (eg. "euler")
		/// \param deviceConfiguration the deviceConfiguration used.
        ///
        CellComputerFactory(const alsfvm::shared_ptr<simulator::SimulatorParameters>& parameters,
							alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration);

        ///
        /// \brief createComputer creates a new cell computer
        /// \return an instance of the cell computer.
        ///
        alsfvm::shared_ptr<CellComputer> createComputer();

    private:
        const alsfvm::shared_ptr<simulator::SimulatorParameters> simulatorParameters;

    };
} // namespace alsfvm
} // namespace equation
