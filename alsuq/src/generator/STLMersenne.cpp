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

#include "alsuq/generator/STLMersenne.hpp"
#include "alsutils/error/Exception.hpp"

namespace alsuq {
namespace generator {
namespace {
// We need to make sure we have a single generator
// so that we generate independent samples
std::pair<std::mt19937_64, long>& getGeneratorInstance() {
    static std::pair<std::mt19937_64, long> generator;

    return generator;
}
}


STLMersenne::STLMersenne(size_t dimension)
    : generator(getGeneratorInstance()), dimension(dimension) {

}

real STLMersenne::generate(size_t component, size_t sample) {


    if (int(component) >= dimension) {
        THROW("Component given higher than dimension. component = "
            << component << ", dimension = " << dimension);
    }

    if (generator.second / dimension < long(sample) - 1) {
        //distribution(generator.first);
        const auto samples_to_be_added = (long(sample) - 1) * dimension -
            generator.second;
        generator.first.discard(samples_to_be_added);
        generator.second += samples_to_be_added;
    }

    if (generator.second % dimension < long(component) - 1) {
        const auto samples_to_be_added = long(component) - 1 - long(generator.second %
                dimension);
        generator.first.discard(samples_to_be_added);
    }

    generator.second++;

    return distribution(generator.first);
}

}
}
