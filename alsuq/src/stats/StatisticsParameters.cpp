#include "alsuq/stats/StatisticsParameters.hpp"

namespace alsuq { namespace stats {

void StatisticsParameters::setNumberOfSamples(size_t samples)
{
    this->samples = samples;
}

size_t StatisticsParameters::getNumberOfSamples() const
{
    return samples;
}

const std::string &StatisticsParameters::getParameterAsString(const std::string &name) const
{
    return configuration.get<std::string>(name);
}

real StatisticsParameters::getParameterAsDouble(const std::string &name) const
{
    return configuration.get<real>(name);
}

void StatisticsParameters::setConfiguration(const boost::property_tree::ptree &configuration)
{
    this->configuration = configuration;
}

}
}
