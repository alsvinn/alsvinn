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

#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/distribution/Normal.hpp"
#include "alsuq/distribution/Uniform.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsuq/distribution/Uniform1D.hpp"
#include "alsuq/distribution/DLLDistribution.hpp"


namespace alsuq {
namespace distribution {

std::shared_ptr<Distribution> DistributionFactory::createDistribution(
    const std::string& name,
    const size_t dimensions,
    const size_t numberVariables,
    const Parameters& parameters) {
    std::shared_ptr<Distribution> distribution;

    if (name == "normal") {
        THROW("Normal variables are not supported in the current implementation."
            << " Use Uniform and scipy.stats.norm.ppf(X) where X is the random viriable.");
        //distribution.reset(new Normal(parameters));
    } else if (name == "uniform") {
        distribution.reset(new Uniform(parameters));
    } else if (name == "uniform1d") {
        distribution.reset(new Uniform1D(numberVariables,
                parameters.getParameter("a"),
                parameters.getParameter("b")));
    } else if (name == "dll") {
        distribution.reset(new DLLDistribution(numberVariables, dimensions,
                parameters));

    } else {
        THROW("Unknown distribution " << name);
    }

    return distribution;
}

}
}
