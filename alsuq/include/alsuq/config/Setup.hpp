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
#include <boost/property_tree/ptree.hpp>
#include "alsuq/samples/SampleGenerator.hpp"
#include "alsuq/mpi/Configuration.hpp"
#include "alsuq/run/Runner.hpp"
#include "alsuq/stats/Statistics.hpp"
namespace alsuq {
namespace config {

class Setup {
public:
    typedef boost::property_tree::ptree ptree;

    //! Creates a new UQ runner.
    //!
    //! @param inputFilename the input XML filename
    //! @param mpiConfigurationWorld the top level mpi configuration to use
    //!                              (for most practical use cases, this is MPI_COMM_WORLD)
    //!
    //! @param multiSample the number of samples to run in parallel
    //!
    //! @param multiSpatial a 3 vector, for which each component is the number of processors to use in each direction.
    //!
    //! @note We require that
    //! \code{.cpp}
    //!   multiSamples*multiSpatial.x*multiSpatial.y*multiSpatial.z == mpiConfigurationWorld.getNumberOfProcesses();
    //! \endcode
    std::shared_ptr<run::Runner> makeRunner(const std::string& inputFilename,
        mpi::ConfigurationPtr mpiConfigurationWorld,
        int multiSample, ivec3 multiSpatial);

private:

    std::shared_ptr<samples::SampleGenerator> makeSampleGenerator(
        ptree& configuration);

    std::vector<std::shared_ptr<stats::Statistics> > createStatistics(
        ptree& configuration,
        alsutils::mpi::ConfigurationPtr statisticalConfiguration,
        mpi::ConfigurationPtr spatialConfiguration,
        mpi::ConfigurationPtr worldConfiguration);
    size_t readNumberOfSamples(ptree& configuration);
};
} // namespace config
} // namespace alsuq
