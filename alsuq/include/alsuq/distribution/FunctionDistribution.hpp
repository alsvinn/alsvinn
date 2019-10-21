#pragma once
#include "alsuq/distribution/Distribution.hpp"
#include <functional>

namespace alsuq {
namespace distribution {

//! Gets a std::function and uses that to generate new samples
class FunctionDistribution : public Distribution {
public:

    FunctionDistribution(std::function<real(size_t, size_t)> distributionFunction);

    virtual real generate(generator::Generator& generator, size_t component,
        size_t sample) override;
private:
    std::function<real(size_t, size_t)> distributionFunction;

};
} // namespace distribution
} // namespace alsuq
