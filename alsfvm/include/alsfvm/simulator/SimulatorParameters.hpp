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

#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/equation/EquationParameters.hpp"
namespace alsfvm {
namespace simulator {

class SimulatorParameters {
public:
    SimulatorParameters()
        : equationParameters(new equation::EquationParameters)
    {}
    SimulatorParameters(const std::string& equationName,
        const std::string& platform);

    void setCFLNumber(real cfl);
    real getCFLNumber() const;

    const equation::EquationParameters& getEquationParameters() const;
    equation::EquationParameters& getEquationParameters();
    void setEquationParameters(alsfvm::shared_ptr<equation::EquationParameters>
        parameters);

    void setEquationName(const std::string& name);

    const std::string& getEquationName() const;

    void setPlatform(const std::string& platform);

    const std::string& getPlatform() const;


private:
    real cflNumber;
    std::string equationName;
    std::string platform;
    alsfvm::shared_ptr<equation::EquationParameters> equationParameters;

};
} // namespace alsfvm
} // namespace simulator
