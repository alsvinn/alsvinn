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
#include "alsuq/distribution/Distribution.hpp"
namespace alsuq {
namespace distribution {

//! Uses the midpoint rule of integration
class Uniform1D : public Distribution {
public:
    Uniform1D(size_t numberOfSamples, real a, real b);
    //! Generates the next random number.
    //! \note ONLY WORKS FOR 1D PROBLEMS
    virtual real generate(generator::Generator& generator, size_t component,
        size_t sample) override;
private:

    real deltaX;
    real a;
};
} // namespace generator
} // namespace alsuq
