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
#include "alsuq/generator/Generator.hpp"
#include <random>

namespace alsuq {
namespace generator {

//! Uses the C++ STL implementation to generate random numbers
//!
//! @todo This needs to be fixed for multiple users (ie. when multiple
//! instances uses this, it will not work properly with the sample fast forwarding)
class STLMersenne : public Generator {
public:
    STLMersenne(size_t dimension);

    //! Generates the next random number
    real generate(size_t component, size_t sample) override;
private:


    std::pair<std::mt19937_64, int>& generator;
    std::uniform_real_distribution<real> distribution{0.0, 1.0};

    const int dimension;
};
} // namespace generator
} // namespace alsuq
