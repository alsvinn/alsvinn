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

#include "alsfvm/init/Parameters.hpp"
#include "alsutils/error/Exception.hpp"
namespace alsfvm {
namespace init {
//! Add a a parameter to the parameters.
void Parameters::addParameter(const std::string& name,
    const std::vector<real>& value) {
    if (parameters.find(name) != parameters.end()) {
        THROW("Parameter already registered: " << name);
    }

    parameters[name] = value;
}


std::vector<std::string> Parameters::getParameterNames() const {
    std::vector<std::string> names;

    for (auto key : parameters) {
        names.push_back(key.first);
    }

    return names;
}

//! Each parameter is represented by an array
//! A scalar is then represented by a length one array.
const std::vector<real>& Parameters::getParameter(const std::string& name)
const {
    if (parameters.find(name) == parameters.end()) {
        THROW("Parameter not found: " << name);
    }

    return parameters.at(name);
}

void Parameters::setOrAddParameter(const std::string& name,
    const std::vector<real>& value) {
    parameters[name] = value;
}
}
}
