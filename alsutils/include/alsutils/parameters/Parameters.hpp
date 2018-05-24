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
#include <boost/property_tree/ptree.hpp>
#include <vector>
#include <map>
namespace alsutils {
namespace parameters {

//! Holds general parameters based on a boost::property tree
class Parameters {
public:
    Parameters(const boost::property_tree::ptree& ptree);

    //! Convenience constructor. Used mostly for unittesting.
    //!
    Parameters(const std::map<std::string, std::string>& values);


    double getDouble(const std::string& name) const;
    int getInteger(const std::string& name) const;
    std::string getString(const std::string& name) const;

    bool contains(const std::string& name) const;

    std::vector<std::string> getStringVectorFromString(const std::string& name)
    const;

    std::vector<std::string> getKeys() const;
private:
    boost::property_tree::ptree ptree;

};
} // namespace parameters
} // namespace alsutils
