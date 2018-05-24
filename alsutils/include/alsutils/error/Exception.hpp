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
#include <sstream>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <boost/current_function.hpp>

///
/// Throws an exception with the given message
///
#define THROW(message) {\
    std::stringstream ssForException; \
    ssForException << message; \
    ssForException << std::endl << "At " << __FILE__<<":" << __LINE__ << std::endl;\
    ssForException << std::endl << "In function: " << BOOST_CURRENT_FUNCTION << std::endl;\
    throw std::runtime_error(ssForException.str());\
}

