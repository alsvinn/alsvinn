#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/CPUCellComputer.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/equation/CUDACellComputer.hpp"
#endif

namespace alsfvm { namespace equation {

CellComputerFactory::CellComputerFactory(const alsfvm::shared_ptr<simulator::SimulatorParameters>& parameters,
										 alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : simulatorParameters(parameters)
{
    // empty
}

alsfvm::shared_ptr<CellComputer> CellComputerFactory::createComputer()
{
    auto platform = simulatorParameters->getPlatform();
    auto equation = simulatorParameters->getEquationName();
    if (platform == "cpu") {
        if (equation == "euler") {
            return alsfvm::shared_ptr<CellComputer>(new CPUCellComputer<euler::Euler>(*simulatorParameters));
        } else {
            THROW("Unknown equation " << equation);
        }
    }
#ifdef ALSVINN_HAVE_CUDA
    if (platform == "cuda") {
		if (equation == "euler") {
            return alsfvm::shared_ptr<CellComputer>(new CUDACellComputer<euler::Euler>(parameters));
		}
		else {
			THROW("Unknown equation " << equation);
		}
	}
#endif
	else {
        THROW("Unknown platform " << platform);
    }
}

}
}
