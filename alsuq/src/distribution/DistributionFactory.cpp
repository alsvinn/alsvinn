#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/distribution/Normal.hpp"
#include "alsuq/distribution/Uniform.hpp"
#include "alsutils/error/Exception.hpp"



namespace alsuq { namespace distribution {

std::shared_ptr<Distribution> DistributionFactory::createDistribution(
        const std::string &name,
        const size_t dimensions,
        const size_t numberVariables,
        const Parameters &parameters)
{
    std::shared_ptr<Distribution> distribution;
    if (name == "normal") {
        distribution.reset(new Normal(parameters));
    } else if(name == "uniform") {
        distribution.reset(new Uniform(parameters));
    } else {
        THROW("Unknown distribution " << name);
    }

    return distribution;
}

}
}
