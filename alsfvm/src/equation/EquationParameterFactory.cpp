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

#include "alsfvm/equation/EquationParameterFactory.hpp"
#include "alsfvm/equation/euler/EulerParameters.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm {
namespace equation {

namespace {
struct EquationParametersFunctor {
    EquationParametersFunctor(const std::string& name,
        alsfvm::shared_ptr<EquationParameters>& parameters)
        : name(name), parameters(parameters) {

    }

    template<class T>
    void operator()(const T& t) const {
        if (T::getName() == name) {
            parameters.reset(new typename T::EquationType::Parameters);
        }
    }

    std::string name;
    alsfvm::shared_ptr<EquationParameters>& parameters;
};
}

alsfvm::shared_ptr<EquationParameters> EquationParameterFactory::createDefaultEquationParameters(
    const std::string& name) {
    alsfvm::shared_ptr<EquationParameters> parameters;
    EquationParametersFunctor equationParametersFunctor(name, parameters);


    for_each_equation(equationParametersFunctor);

    if (!parameters) {
        THROW("Unknown equation " << name);
    }

    return parameters;
}

}
}
