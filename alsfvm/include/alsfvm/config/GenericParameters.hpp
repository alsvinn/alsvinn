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
namespace alsfvm {
namespace config {

//! Base class to hold parameters through a boost property tree. To be
//! passed around to other classes.
class GenericParameters {
public:
    GenericParameters(const boost::property_tree::ptree& tree);

    //! Gets a double parameter with key key
    double getDouble(const std::string& key) const;

    //! Gets an integer parameter with key key
    int getInteger(const std::string& key) const;


    //! Gets a double parameter with key key,
    //!
    //! if they key does not exist, returns defaultValue
    double getDouble(const std::string& key, double defaultValue) const;

    //! Gets an integer parameter with key key
    //!
    //! if they key does not exist, returns defaultValue
    int getInteger(const std::string& key, int defaultValue) const;
private:
    boost::property_tree::ptree tree;
};
} // namespace config
} // namespace alsfvm
