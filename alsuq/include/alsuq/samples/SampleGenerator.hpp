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
#include "alsuq/generator/Generator.hpp"
#include "alsuq/distribution/Distribution.hpp"
#include <map>
#include <string>

namespace alsuq {
namespace samples {

class SampleGenerator {
public:
    typedef  std::map<std::string, std::pair<size_t,
             std::pair<
             std::shared_ptr<generator::Generator>,
             std::shared_ptr<distribution::Distribution> > > >
             GeneratorDistributionMap;


    SampleGenerator(const GeneratorDistributionMap& generators);


    std::vector<real> generate(const std::string& parameter,
        const size_t sampleIndex);

    std::vector<std::string> getParameterList() const;
private:

    GeneratorDistributionMap generators;
};
} // namespace samples
} // namespace alsuq
