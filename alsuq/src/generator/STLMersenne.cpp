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
#include "alsutils/log.hpp"

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

    if (long(component) >= long(dimension)) {
        THROW("Component given higher than dimension. component = "
            << component << ", dimension = " << dimension);
    }


    if (generator.second / long(dimension) < long(sample) - 1) {


        const auto samplesToBeAdded = (long(sample) - 1) * dimension -
            generator.second;

        ALSVINN_LOG(INFO, "(not matching sample) Discarding " << samplesToBeAdded <<
            " samples"
            << ", generatedSamples = " << generator.second);
        generator.first.discard(samplesToBeAdded);
        ALSVINN_LOG(INFO, "Done discarding " << samplesToBeAdded
            << " samples:(\n\tsample = " << sample
            << "\n\tcomponent = " << component
            << "\n\tdimension = " << dimension
            << "\n\tgenerator.second / long(dimension) = " << generator.second /
            long(dimension) << "\n)");
        generator.second += samplesToBeAdded;
    }

    if (generator.second % long(dimension) < long(component) - 1) {
        const auto samplesToBeAdded = long(component) - 1 - long(generator.second %
                dimension);
        ALSVINN_LOG(INFO, "(not matching component) Discarding " << samplesToBeAdded <<
            " samples"
            << ", generatedSamples = " << generator.second);
        generator.first.discard(samplesToBeAdded);
        ALSVINN_LOG(INFO, "Done discarding " << samplesToBeAdded
            << " samples (\n\tsample = " << sample
            << "\n\tcomponent = " << component
            << "\n\tdimension = " << dimension
            << "\n\tgenerator.second % long(dimension) = " << (generator.second %
                long(dimension)) << "\n)");

        generator.second += samplesToBeAdded;
    }

    generator.second++;

    return distribution(generator.first);
}

}
}
