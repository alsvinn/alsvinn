#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/CPUCellComputer.hpp"
#ifdef ALSVINN_HAVE_CUDA
#include "alsfvm/equation/CUDACellComputer.hpp"
#endif

namespace alsfvm { namespace equation {

CellComputerFactory::CellComputerFactory(const std::string &platform,
                                         const std::string &equation,
										 boost::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : platform(platform), equation(equation)
{
    // empty
}

boost::shared_ptr<CellComputer> CellComputerFactory::createComputer()
{
    if (platform == "cpu") {
        if (equation == "euler") {
            return boost::shared_ptr<CellComputer>(new CPUCellComputer<euler::Euler>());
        } else {
            THROW("Unknown equation " << equation);
        }
    }
#ifdef ALSVINN_HAVE_CUDA
    if (platform == "cuda") {
		if (equation == "euler") {
			return boost::shared_ptr<CellComputer>(new CUDACellComputer<euler::Euler>());
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
