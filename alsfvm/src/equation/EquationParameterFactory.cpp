#include "alsfvm/equation/EquationParameterFactory.hpp"
#include "alsfvm/equation/euler/EulerParameters.hpp"
#include "alsfvm/error/Exception.hpp"

namespace alsfvm { namespace equation {

alsfvm::shared_ptr<EquationParameters> EquationParameterFactory::createDefaultEquationParameters(const std::string &name)
{
    if (name == "euler") {
        return alsfvm::shared_ptr<EquationParameters>(new euler::EulerParameters());
    }
    THROW("Unknown equation " << name);
}

}
}
