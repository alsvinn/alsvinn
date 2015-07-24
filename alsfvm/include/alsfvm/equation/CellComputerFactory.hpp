#pragma once
#include <memory>
#include "alsfvm/equation/CellComputer.hpp"

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
        ///
        CellComputerFactory(const std::string& platform,
                            const std::string& equation);

        ///
        /// \brief createComputer creates a new cell computer
        /// \return an instance of the cell computer.
        ///
        std::shared_ptr<CellComputer> createComputer();

    private:
        const std::string platform;
        const std::string equation;

    };
} // namespace alsfvm
} // namespace equation
