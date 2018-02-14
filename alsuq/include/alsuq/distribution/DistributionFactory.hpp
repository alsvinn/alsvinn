#pragma once
#include "alsuq/distribution/Distribution.hpp"
#include "alsuq/distribution/Parameters.hpp"
#include "alsuq/types.hpp"

namespace alsuq {
namespace distribution {

class DistributionFactory {
    public:
        std::shared_ptr<Distribution> createDistribution(const std::string& name,
            const size_t dimensions,
            const size_t numberVariables,
            const Parameters& parameters);
};
} // namespace distribution
} // namespace alsuq
