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
#include "alsutils/parameters/Parameters.hpp"
#include "alsuq/types.hpp"
#include <boost/property_tree/ptree.hpp>
#include "alsuq/mpi/Configuration.hpp"
namespace alsuq {
namespace stats {

class StatisticsParameters : public alsutils::parameters::Parameters {
public:
    StatisticsParameters(const boost::property_tree::ptree& configuration);

    void setNumberOfSamples(size_t samples);
    size_t getNumberOfSamples() const;

    mpi::ConfigurationPtr getMpiConfiguration() const;
    void setMpiConfiguration(mpi::ConfigurationPtr value);

    void setPlatform(const std::string& platform);
    std::string getPlatform() const;
private:

    size_t samples =  0;

    mpi::ConfigurationPtr mpiConfiguration = nullptr;

    std::string platform = "cpu";
};
} // namespace stats
} // namespace alsuq
