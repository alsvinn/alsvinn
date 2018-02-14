#pragma once
#include "alsfvm/equation/EquationParameters.hpp"
#include "alsfvm/types.hpp"
namespace alsfvm {
namespace equation {

class EquationParameterFactory {
public:
    alsfvm::shared_ptr<EquationParameters> createDefaultEquationParameters(
        const std::string& name);
};
} // namespace alsfvm
} // namespace equation
