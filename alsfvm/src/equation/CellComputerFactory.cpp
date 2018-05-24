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

#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/CPUCellComputer.hpp"
#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/equation/CUDACellComputer.hpp"
#endif

#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm {
namespace equation {

namespace {
struct CellComputerFunctor {
    CellComputerFunctor( simulator::SimulatorParameters& parameters,
        alsfvm::shared_ptr<CellComputer>& cellComputerPointer)
        : parameters(parameters), cellComputerPointer(cellComputerPointer) {

    }

    template<class EquationInfo>
    void operator()(const EquationInfo& t) const {
        auto platform = parameters.getPlatform();

        if (t.getName() == parameters.getEquationName()) {
            if (platform == "cpu") {
                cellComputerPointer.reset(
                    new CPUCellComputer<typename EquationInfo::EquationType>(parameters));
            }

#ifdef ALSVINN_HAVE_CUDA
            else if (platform == "cuda") {
                cellComputerPointer.reset(
                    new CUDACellComputer<typename EquationInfo::EquationType>(parameters));
            }

#endif
            else {
                THROW("Unknown platform " << platform);
            }
        }
    }

    simulator::SimulatorParameters& parameters;
    alsfvm::shared_ptr<CellComputer>& cellComputerPointer;

};
}

CellComputerFactory::CellComputerFactory(const
    alsfvm::shared_ptr<simulator::SimulatorParameters>& parameters,
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : simulatorParameters(parameters) {
    // empty
}

alsfvm::shared_ptr<CellComputer> CellComputerFactory::createComputer() {
    alsfvm::shared_ptr<CellComputer> cellComputerPointer;

    for_each_equation(CellComputerFunctor(*simulatorParameters,
            cellComputerPointer));

    if (!cellComputerPointer) {
        THROW("Unrecognized equation " << simulatorParameters->getEquationName());
    }

    return cellComputerPointer;

}

}
}
