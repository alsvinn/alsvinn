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

#include "alsutils/parameters/Parameters.hpp"
#include <boost/algorithm/string.hpp>
namespace alsutils {
namespace parameters {

Parameters::Parameters(const boost::property_tree::ptree& ptree)
    : ptree(ptree) {

}

Parameters::Parameters(const std::map<std::string, std::string>& values) {
    for (const auto& p : values) {
        ptree.add(p.first, p.second);
    }
}

double Parameters::getDouble(const std::string& name) const {
    return ptree.get<double>(name);
}

int Parameters::getInteger(const std::string& name) const {
    return ptree.get<int>(name);
}

std::string Parameters::getString(const std::string& name) const {
    return ptree.get<std::string>(name);
}

bool Parameters::contains(const std::string& name) const {
    return ptree.find(name) != ptree.not_found();
}

std::vector<std::string> Parameters::getStringVectorFromString(
    const std::string& name) const {
    std::vector<std::string> strings;
    std::string inputString = ptree.get<std::string>(name);

    boost::split(strings, inputString, boost::is_any_of(" \t"));

    return strings;
}

std::vector<std::string> Parameters::getKeys() const {
    std::vector<std::string> parameters;

    for (const auto& child : ptree) {
        parameters.push_back(child.first);
    }

    return parameters;
}

void Parameters::addIntegerParameter(const std::string& name, int i) {
    ptree.put(name, i);
}

void Parameters::addStringParameter(const std::string& name,
    std::string value) {
    ptree.put(name, value);
}

void Parameters::addVectorParameter(const std::string& name,
    const std::vector<std::string>& values) {
    auto valuesCopy = values;
    ptree.put(name, boost::algorithm::join(valuesCopy, " "));
}

Parameters Parameters::empty() {
    boost::property_tree::ptree ptree;
    return Parameters(ptree);
}

}
}
