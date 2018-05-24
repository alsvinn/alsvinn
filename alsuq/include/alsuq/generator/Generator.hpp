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
#include "alsuq/types.hpp"
#include <boost/noncopyable.hpp>

namespace alsuq {
namespace generator {

//! A generator is used to generate random numbers,
//! use a distribution to get them under some distribution
//!
//! \note All generators are singletons
class Generator : public boost::noncopyable {
public:
    virtual ~Generator() {};

    //! Generates a uniformly distributed number between 0 and 1
    virtual real generate(size_t component) = 0;
};
} // namespace generator
} // namespace alsuq
