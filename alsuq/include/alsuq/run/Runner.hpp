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
#include "alsuq/samples/SampleGenerator.hpp"
#include "alsuq/run/SimulatorCreator.hpp"
#include "alsuq/stats/Statistics.hpp"

namespace alsuq {
namespace run {

class Runner {
public:
    Runner(std::shared_ptr<SimulatorCreator> simulatorCreator,
        std::shared_ptr<samples::SampleGenerator> sampleGenerator,
        std::vector<size_t> sampleNumbers,
        mpi::ConfigurationPtr mpiConfig,
        const std::string& name);




    void run();


    //! Sets the statistics to be used
    void setStatistics(const std::vector<std::shared_ptr<stats::Statistics> >&
        statistics);
    std::string getName() const;

    size_t getTimestepsPerformedTotal() const;

private:
    std::shared_ptr<SimulatorCreator> simulatorCreator;
    std::shared_ptr<samples::SampleGenerator> sampleGenerator;
    std::vector<std::string> parameterNames;
    std::vector<size_t> sampleNumbers;
    std::vector<std::shared_ptr<stats::Statistics> > statistics;

    mpi::ConfigurationPtr mpiConfig;
    const std::string name;
    size_t timestepsPerformedTotal = 0;
};
} // namespace run
} // namespace alsuq
