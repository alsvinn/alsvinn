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

#include "alsuq/distribution/Uniform1D.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsutils/log.hpp"
#include <iostream>
namespace alsuq {
namespace distribution {
Uniform1D::Uniform1D(size_t numberOfSamples, real a, real b)
    : a(a) {
    ALSVINN_LOG(INFO, "numberOfSampes = " << numberOfSamples);

    deltaX = (b - a) / numberOfSamples;
}

real Uniform1D::generate(generator::Generator& generator, size_t component) {
    if (component > 0) {
        THROW("With Uniform1D we only support 1 dimension");
    }

    real midpoint = a + deltaX * (currentSample + 0.5);
    currentSample++;
    return midpoint;
}

}
}
