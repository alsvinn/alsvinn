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

#include "alsfvm/config/GenericParameters.hpp"

namespace alsfvm {
namespace config {

GenericParameters::GenericParameters(const boost::property_tree::ptree& tree)
    : tree(tree) {

}

double GenericParameters::getDouble(const std::string& key) const {
    return tree.get<double>(key);
}

int GenericParameters::getInteger(const std::string& key) const {
    return tree.get<int>(key);
}

double GenericParameters::getDouble(const std::string& key,
    double defaultValue) const {
    if (tree.find(key) != tree.not_found()) {
        return getDouble(key);
    } else {
        return defaultValue;
    }
}

int GenericParameters::getInteger(const std::string& key,
    int defaultValue) const {
    if (tree.find(key) != tree.not_found()) {
        return getInteger(key);
    } else {
        return defaultValue;
    }
}

}
}
