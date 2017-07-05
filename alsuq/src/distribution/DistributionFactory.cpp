#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/distribution/Normal.hpp"
#include "alsuq/distribution/Uniform.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsuq/distribution/Uniform1D.hpp"


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
    } else if(name == "uniform1d") {
	distribution.reset(new Uniform1D(numberVariables,
				     parameters.getParameter("a"),
				     parameters.getParameter("b")));
    }
    else {
        THROW("Unknown distribution " << name);
    }

    return distribution;
}

}
}
