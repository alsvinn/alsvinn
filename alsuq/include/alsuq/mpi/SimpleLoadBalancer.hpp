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
#include "alsuq/mpi/Configuration.hpp"
#include <vector>
#include "alsuq/types.hpp"

namespace alsuq {
namespace mpi {

class SimpleLoadBalancer {
public:
    SimpleLoadBalancer(const std::vector<size_t>& samples);

    //! @param multiSample the number of samples to run in parallel
    //!
    //! @param multiSpatial a 3 vector, for which each component is the number of processors to use in each direction.
    //!
    //! @note We require that
    //! \code{.cpp}
    //!   multiSample*multiSpatial.x*multiSpatial.y*multiSpatial.z == mpiConfigurationWorld.getNumberOfProcesses();
    //! \endcode
    //!
    //! @param mpiConfig the relevant mpiConfig
    //!
    //! \return a tuple, where the first component is the list of samples to compute, the second is the configuration of the statistical domain,
    //! and the last is the configuration of the parallel domain.
    //!
    std::tuple<std::vector<size_t>, ConfigurationPtr, ConfigurationPtr> loadBalance(
        int multiSample, ivec3 multiSpatial,
        const Configuration& mpiConfig);

private:
    std::vector<size_t> samples;
};
} // namespace mpi
} // namespace alsuq
