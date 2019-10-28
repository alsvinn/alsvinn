#include "alsuq/distribution/FunctionDistribution.hpp"

namespace alsuq {
namespace distribution {

FunctionDistribution::FunctionDistribution(std::function<real (size_t, size_t)>
    distributionFunction)
    : distributionFunction(distributionFunction) {

}

real FunctionDistribution::generate(generator::Generator& generator,
    size_t component, size_t sample) {
    return distributionFunction(component, sample);
}

}
}
