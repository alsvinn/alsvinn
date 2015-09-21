#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/equation/CPUCellComputer.hpp"
#include "alsfvm/equation/CUDACellComputer.hpp"

namespace alsfvm { namespace equation {

CellComputerFactory::CellComputerFactory(const std::string &platform,
                                         const std::string &equation,
										 std::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : platform(platform), equation(equation)
{
    // empty
}

std::shared_ptr<CellComputer> CellComputerFactory::createComputer()
{
    if (platform == "cpu") {
        if (equation == "euler") {
            return std::shared_ptr<CellComputer>(new CPUCellComputer<euler::Euler>());
        } else {
            THROW("Unknown equation " << equation);
        }
	} if (platform == "cuda") {
		if (equation == "euler") {
			return std::shared_ptr<CellComputer>(new CUDACellComputer<euler::Euler>());
		}
		else {
			THROW("Unknown equation " << equation);
		}
	}
	else {
        THROW("Unknown platform " << platform);
    }
}

}
}
