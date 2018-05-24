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

#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/equation/EquationParameterFactory.hpp"
namespace alsfvm {
namespace simulator {

SimulatorParameters::SimulatorParameters(const std::string& equationName,
    const std::string& platform)
    : equationName(equationName), platform(platform) {
    equation::EquationParameterFactory factory;
    equationParameters = factory.createDefaultEquationParameters(equationName);
}

void SimulatorParameters::setCFLNumber(real cfl) {
    cflNumber = cfl;
}

real SimulatorParameters::getCFLNumber() const {
    return cflNumber;
}

const equation::EquationParameters& SimulatorParameters::getEquationParameters()
const {
    return *equationParameters;
}

equation::EquationParameters& SimulatorParameters::getEquationParameters() {
    return *equationParameters;
}


void SimulatorParameters::setEquationParameters(
    alsfvm::shared_ptr<equation::EquationParameters> parameters) {
    equationParameters = parameters;
}

void SimulatorParameters::setEquationName(const std::string& name) {
    equationName = name;
}

const std::string& SimulatorParameters::getEquationName() const {
    return equationName;
}

void SimulatorParameters::setPlatform(const std::string& platform) {
    this->platform = platform;
}

const std::string& SimulatorParameters::getPlatform() const {
    return platform;
}

}
}
