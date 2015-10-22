#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/equation/EquationParameterFactory.hpp"
namespace alsfvm { namespace simulator {

SimulatorParameters::SimulatorParameters(const std::string &equationName, const std::string &platform)
    :equationName(equationName), platform(platform)
{
    equation::EquationParameterFactory factory;
    equationParameters = factory.createDefaultEquationParameters(equationName);
}

void SimulatorParameters::setCFLNumber(real cfl)
{
    cflNumber = cfl;
}

real SimulatorParameters::getCFLNumber() const
{
    return cflNumber;
}

const equation::EquationParameters &SimulatorParameters::getEquationParameters() const
{
    return *equationParameters;
}

equation::EquationParameters &SimulatorParameters::getEquationParameters()
{
    return *equationParameters;
}


void SimulatorParameters::setEquationParameters(alsfvm::shared_ptr<equation::EquationParameters> parameters)
{
    equationParameters = parameters;
}

void SimulatorParameters::setEquationName(const std::string &name)
{
    equationName = name;
}

const std::string &SimulatorParameters::getEquationName() const
{
    return equationName;
}

void SimulatorParameters::setPlatform(const std::string &platform)
{
    this->platform = platform;
}

const std::string &SimulatorParameters::getPlatform() const
{
    return platform;
}

}
}
