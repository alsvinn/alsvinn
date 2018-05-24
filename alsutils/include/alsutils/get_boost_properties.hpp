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
#include <boost/version.hpp>
#include <boost/config.hpp>
namespace alsutils {
inline boost::property_tree::ptree getBoostProperties() {
    boost::property_tree::ptree properties;
    properties.put("BOOST_VERSION", BOOST_VERSION);
    properties.put("BOOST_LIB_VERSION", BOOST_LIB_VERSION);
    properties.put("BOOST_PLATFORM", BOOST_PLATFORM);
    properties.put("BOOST_PLATFORM_CONFIG", BOOST_PLATFORM_CONFIG);
    properties.put("BOOST_COMPILER", BOOST_COMPILER);
    #ifdef BOOST_LIBSTDCXX_VERSION
    properties.put("BOOST_LIBSTDCXX_VERSION", BOOST_LIBSTDCXX_VERSION);
    #endif
    #ifdef BOOST_LIBSTDCXX11
    properties.put("C++11", true);
#else
    properties.add("C++11", false);
    #endif
    properties.put("BOOST_STDLIB", BOOST_STDLIB);
    properties.put("BOOST_STDLIB_CONFIG", BOOST_STDLIB_CONFIG);

    return properties;
}
}
