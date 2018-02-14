#pragma once
#include "alsuq/mpi/Configuration.hpp"
#include <vector>
#include "alsuq/types.hpp"

namespace alsuq {
namespace mpi {

class SimpleLoadBalancer {
    public:
        SimpleLoadBalancer(const std::vector<size_t>& samples);

        //! @param multiSamples the number of samples to run in parallel
        //!
        //! @param multiSpatial a 3 vector, for which each component is the number of processors to use in each direction.
        //!
        //! @note We require that
        //! \code{.cpp}
        //!   multiSamples*multiSpatial.x*multiSpatial.y*multiSpatial.z == mpiConfigurationWorld.getNumberOfProcesses();
        //! \endcode
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
