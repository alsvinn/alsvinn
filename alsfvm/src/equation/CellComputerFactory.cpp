#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/CPUCellComputer.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/equation/CUDACellComputer.hpp"
#endif

#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm { namespace equation {

namespace {
struct CellComputerFunctor {
    CellComputerFunctor( simulator::SimulatorParameters& parameters,
                        alsfvm::shared_ptr<CellComputer>& cellComputerPointer)
        : parameters(parameters), cellComputerPointer(cellComputerPointer)
    {

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

CellComputerFactory::CellComputerFactory(const alsfvm::shared_ptr<simulator::SimulatorParameters>& parameters,
										 alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : simulatorParameters(parameters)
{
    // empty
}

alsfvm::shared_ptr<CellComputer> CellComputerFactory::createComputer()
{
    alsfvm::shared_ptr<CellComputer> cellComputerPointer;

    for_each_equation(CellComputerFunctor(*simulatorParameters, cellComputerPointer));

    if (!cellComputerPointer) {
        THROW("Unrecognized equation " << simulatorParameters->getEquationName());
    }

    return cellComputerPointer;

}

}
}
