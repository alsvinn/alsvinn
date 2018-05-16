#include "alsuq/stats/StatisticsParameters.hpp"

namespace alsuq {
namespace stats {

StatisticsParameters::StatisticsParameters(const boost::property_tree::ptree&
    configuration)
    : alsutils::parameters::Parameters(configuration) {

}

void StatisticsParameters::setNumberOfSamples(size_t samples) {
    this->samples = samples;
}

size_t StatisticsParameters::getNumberOfSamples() const {
    return samples;
}

mpi::ConfigurationPtr StatisticsParameters::getMpiConfiguration() const {
    return mpiConfiguration;
}

void StatisticsParameters::setMpiConfiguration(mpi::ConfigurationPtr value) {
    mpiConfiguration = value;
}

void StatisticsParameters::setPlatform(const std::string& platform) {
    this->platform = platform;
}

std::string StatisticsParameters::getPlatform() const {
    return platform;
}

}
}
