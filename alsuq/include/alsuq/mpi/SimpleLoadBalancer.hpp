#pragma once
#include "alsuq/mpi/Config.hpp"
#include <vector>
namespace alsuq { namespace mpi { 

    class SimpleLoadBalancer {
    public:
        SimpleLoadBalancer(const std::vector<size_t> samples);
        std::vector<size_t> getSamples(const Config& mpiConfig);

    private:
        std::vector<size_t> samples;
    };
} // namespace mpi
} // namespace alsuq
