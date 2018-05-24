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

#include "alsuq/stats/StatisticsParameters.hpp"

namespace alsuq {
namespace stats {

StatisticsParameters::StatisticsParameters(const boost::property_tree::ptree&
    configuration)
    : alsutils::parameters::Parameters(configuration) {

}

void StatisticsParameters::setNumberOfSamples(size_t samples) {
    this->samples = samples;
}

size_t StatisticsParameters::getNumberOfSamples() const {
    return samples;
}

mpi::ConfigurationPtr StatisticsParameters::getMpiConfiguration() const {
    return mpiConfiguration;
}

void StatisticsParameters::setMpiConfiguration(mpi::ConfigurationPtr value) {
    mpiConfiguration = value;
}

void StatisticsParameters::setPlatform(const std::string& platform) {
    this->platform = platform;
}

std::string StatisticsParameters::getPlatform() const {
    return platform;
}

}
}
